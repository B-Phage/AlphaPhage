"""
DuckDB I/O utilities for HMM-based annotation pipelines.

This module provides functions for persisting HMM scan results and
associated protein sequences into DuckDB tables. It acts as an I/O
boundary between HMM analysis (e.g. pyhmmer scans) and downstream
processing steps such as multiple sequence alignment (MSA).

Responsibilities:
- Writing HMM scan outputs a parquet lake
- Joining HMM hits with protein sequence data
- Creating tables on demand when they do not yet exist

Non-responsibilities:
- Performing HMM scans
- Running multiple sequence alignments
- Implementing biological domain logic

The functions in this module assume that upstream HMM analysis has
already been performed and that input data is provided in tabular
(pandas DataFrame) and pyhmmer easel formats.
"""

from pathlib import Path
from typing import Iterable, List

import duckdb
import pandas as pd
import psutil
import utils.consol_log as log
from data_contracts.hmm_lake import (
    HMM_LAKE_PRIMARY_KEYS,
    HMM_LAKE_SCHEMA,
)
from hmm.search import normalize_name
from hmm.types import Protein
from storage_io.duckdb_client import DuckDBClient
from storage_io.parquet_lake import enforce_schema, write_parquet_partition

logger = log.get_logger()


def get_safe_duckdb_memory_limit(fraction: float = 0.75) -> str:
    """
    Calculates a safe memory limit in GB based on total system RAM.
    Defaults to 75% of total RAM.
    """
    total_ram_bytes = psutil.virtual_memory().total
    # Convert bytes to GB and apply the fraction
    total_ram_gb = total_ram_bytes / (1024**3)
    safe_ram_gb = int(total_ram_gb * fraction)
    logger.info(f"The total RAM is {total_ram_gb}GB. Setting DuckDB safe limit to {safe_ram_gb}GB")
    return f"{safe_ram_gb}GB"


def write_hmm_results_with_sequences(
    hmm_scan_df: pd.DataFrame,
    protein_py_easel_obj: Iterable[Protein],
    organism_name: str,
    out_root: Path,
):
    """
    Write HMM scan results and associated protein sequences to a DuckDB table.

    This function persists the core outputs of an HMM scan by combining:
    1) tabular HMM hit results (as a pandas DataFrame), and
    2) protein sequences obtained from a pyhmmer easel SequenceFile block.

    Protein sequences are outer-joined to the HMM results on `protein_name`,
    ensuring that proteins with no HMM hits are still written to the table.

    If the target table does not already exist, it will be created with a
    predefined schema. Data is inserted using a composite primary key
    (`protein_name`, `organism_name`, `hmm_domain_name`) to support idempotent
    or incremental writes.

    Parameters
    ----------
    hmm_scan_df : pandas.DataFrame
        DataFrame containing HMM scan results. Must include the following
        columns:
            - protein_name
            - organism_name
            - hmm_domain_name
            - annot
            - evalue
            - score
            - bias
            - length
            - top_hmm_domain_hit_on_prot
            - top_protein_match_for_hmm
            - best_dom_env_start
            - best_dom_env_end
            - best_dom_env_length

    protein_py_easel_obj : iterable
        Iterable of pyhmmer easel sequence objects (e.g. a block returned by
        `pyhmmer.easel.SequenceFile.read_block()`), providing protein names and
        amino acid sequences.

    organism_name : str
        Name of the organism (usually the fasta name) to write to the data lake.

    out_root : Path
        Data Lake's root path.

    Returns
    -------
    None

    Notes
    -----
    This function performs no validation or filtering of HMM results and
    assumes all upstream HMM analysis has already been completed. It is
    intended as an I/O boundary between HMM scanning and downstream workflows
    such as multiple sequence alignment (MSA).
    """
    if hmm_scan_df.empty:
        raise ValueError(
            f"No HMM hits to insert for {organism_name}. Check if the input protein file {organism_name}.fa is corrupted."
        )
    else:
        df_for_sql = hmm_scan_df.copy()

    out_path = out_root / f"organism={organism_name}" / "proteins.parquet"
    logger.info(f"Writing HMM hits to data lake '{out_path}'")

    query_prot = []
    protein_sequence = []
    for prot in protein_py_easel_obj:
        name = normalize_name(prot.name)
        sequence = prot.textize().sequence
        query_prot.append(name)
        protein_sequence.append(sequence)

    # Create the DataFrame
    data = {"protein_name": query_prot, "sequence": protein_sequence}
    seq_df = pd.DataFrame(data)
    seq_df["organism_name"] = organism_name
    df_for_sql = pd.merge(df_for_sql, seq_df, how="outer", on=["protein_name", "organism_name"])
    df_for_sql["hmm_domain_name"] = df_for_sql["hmm_domain_name"].fillna("-")

    # final checks
    df_for_sql = enforce_schema(df_for_sql, HMM_LAKE_SCHEMA)
    write_parquet_partition(df_for_sql, out_path, HMM_LAKE_PRIMARY_KEYS)


def build_hmm_summary_table(
    db_path: Path,
    lake_root: Path,
    table_name: str,
    min_score: float,
) -> None:
    """
    Fully regenerate a DuckDB summary table from the parquet HMM lake.

    This function implements the "Single Best Representative" logic to ensure
    each HMM is mapped to the most biologically relevant protein in an organism.

    Filtering Logic:
    ----------------
    1. HMM-to-Protein Selection: We only keep a pair if it is the absolute best
       match for that HMM across the entire organism
       (top_protein_match_for_hmm = 1).

    2. Domain Conflict Resolution: Once an HMM has identified its best protein,
       we retain the record only if:
       - It is the primary domain hit for that protein (top_hmm_domain_hit_on_prot = 1).
       - OR it is a secondary hit that still carries significant signal
         (score > min_score).

    Example:
        If a protein hits both 'wzy_C' and 'DUF2000', and DUF2000 has the
        higher score, it is the 'primary' hit. We still include the 'wzy_C'
        annotation for this protein if 'wzy_C' has no better match elsewhere
        and its score is > min_score.
    """

    con = duckdb.connect(str(db_path))
    mem_limit = get_safe_duckdb_memory_limit(0.75)

    con.execute(f"SET max_memory='{mem_limit}';")
    con.execute("SET threads TO 4;")  # Prevents too many simultaneous allocations

    parquet_glob = str(lake_root / "organism=*" / "proteins.parquet")

    # Use 'CREATE OR REPLACE' for a more efficient disk-streaming operation
    query = f"""
        CREATE OR REPLACE TABLE {table_name} AS
        SELECT
            protein_name,
            organism_name,
            hmm_domain_name,
            annot,
            sequence
        FROM read_parquet('{parquet_glob}', hive_partitioning=1)
        WHERE
            top_protein_match_for_hmm = 1
            AND (
                top_hmm_domain_hit_on_prot = 1
                OR score > {min_score}
            );
    """

    try:
        con.execute(query)
    finally:
        con.close()


def get_distinct_hmms(
    db_path: Path,
    table_name: str,
    min_copies: int = 2,
) -> List[str]:
    """
    Return a sorted list of distinct HMM domain names from a summary table
    that appear at least `min_copies` times.
    This is intended for orchestrators to parallelise downstream work
    (e.g. exporting FASTA files per HMM).

    Parameters
    ----------
    db_path : Path
        Path to the DuckDB database.
    table_name : str
        Name of the summary table.
    min_copies : int
        Minimum number of occurrences required to include an HMM. Defaults to 2.

    Returns
    -------
    List[str]
        Sorted list of unique hmm_domain_name values meeting the threshold.
    """

    con = duckdb.connect(str(db_path), read_only=True)

    # Use GROUP BY and HAVING to filter by count
    query = f"""
        SELECT hmm_domain_name
        FROM {table_name}
        GROUP BY hmm_domain_name
        HAVING COUNT(*) >= ?
        ORDER BY hmm_domain_name
    """

    # Using a parameterized query (?) is safer and handles the integer injection
    results = con.execute(query, [min_copies]).fetchall()
    hmms = [row[0] for row in results]

    con.close()
    return hmms


def extract_hmm_sequences_for_msa(
    db_path: Path,
    table_name: str,
    hmm_domain_name: str,
    out_fasta: Path,
) -> None:
    """
    Export all proteins matching a given HMM from a DuckDB summary table to a FASTA file.
    These will be used to construct an MSA for the HMM.

    FASTA header format:
        >organism_name;protein_name

    Parameters
    ----------
    db_path : Path
        Path to the DuckDB database file.
    table_name : str
        Name of the summary table (e.g. "hmm_summary").
    hmm_domain_name : str
        HMM domain to filter on.
    out_fasta : Path
        Output FASTA file path.
    """

    con = duckdb.connect(str(db_path), read_only=True)

    query = f"""
        SELECT
            protein_name,
            organism_name,
            sequence
        FROM {table_name}
        WHERE hmm_domain_name = ?
        ORDER BY organism_name, protein_name
    """

    out_fasta.parent.mkdir(parents=True, exist_ok=True)

    with open(out_fasta, "w") as fasta_file:
        for protein_name, organism_name, sequence in con.execute(
            query, [hmm_domain_name]
        ).fetchall():
            fasta_file.write(f">{organism_name};{protein_name}\n")
            fasta_file.write(f"{sequence}\n")

    con.close()


# ------------------------
# Hash / Fingerprint Functions
# ------------------------


def fingerprint_hmm_table(con: duckdb.DuckDBPyConnection, table: str) -> str:
    """Get a deterministic fingerprint of the table contents."""

    row = con.execute(
        f"""
        SELECT sha256(string_agg(
            protein_name || organism_name || hmm_domain_name || sequence,
            ''
            ORDER BY protein_name, organism_name, hmm_domain_name
        ))
        FROM {table};
    """
    ).fetchone()

    # row could be None or row[0] could be None if table is empty
    if row is None or row[0] is None:
        return "0" * 64  # or some sentinel hash for empty table

    return row[0]
