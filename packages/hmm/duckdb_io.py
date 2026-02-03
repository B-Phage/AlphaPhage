"""
DuckDB I/O utilities for HMM-based annotation pipelines.

This module provides functions for persisting HMM scan results and
associated protein sequences into DuckDB tables. It acts as an I/O
boundary between HMM analysis (e.g. pyhmmer scans) and downstream
processing steps such as multiple sequence alignment (MSA).

Responsibilities:
- Writing HMM scan outputs to DuckDB
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

import hashlib
from pathlib import Path
from typing import Iterable, List

import duckdb
import pandas as pd
import utils.consol_log as log
from data_contracts.hmm_lake import (
    HMM_LAKE_PRIMARY_KEYS,
    HMM_LAKE_SCHEMA,
)
from hmm.search import normalize_name
from hmm.types import Protein
from storage_io.duckdb_client import DuckDBClient
from storage_io.parquet_lake import enforce_schema, write_parquet_partition

logger = log.get_logger(level=log.DEBUG)


def hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def hash_lake(lake_root: Path) -> str:
    """Hash all parquet files deterministically."""
    hashes = []
    for p in sorted(lake_root.rglob("*.parquet")):
        hashes.append(hash_file(p))
    return hashlib.sha256("".join(hashes).encode()).hexdigest()


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
    Create or refresh a DuckDB summary table from the parquet HMM lake.

    The table contains only proteins that:
        - are the top protein match for an HMM
        - and either the top domain hit on the protein OR score > min_score

    The table is safe to rebuild as the lake grows.
    """

    con = duckdb.connect(str(db_path))

    parquet_glob = str(lake_root / "organism=*" / "proteins.parquet")

    # Create table if not exists
    con.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            protein_name VARCHAR,
            organism_name VARCHAR,
            hmm_domain_name VARCHAR,
            annot VARCHAR,
            sequence VARCHAR,
            PRIMARY KEY (protein_name, organism_name, hmm_domain_name)
        );
    """
    )

    # Insert new rows from the lake, respecting the PK
    con.execute(
        f"""
        INSERT OR REPLACE INTO {table_name}
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
    )

    con.close()


def get_distinct_hmms(
    db_path: Path,
    table_name: str,
) -> List[str]:
    """
    Return a sorted list of distinct HMM domain names from a summary table.

    This is intended for orchestrators to parallelise downstream work
    (e.g. exporting FASTA files per HMM).

    Parameters
    ----------
    db_path : Path
        Path to the DuckDB database.
    table_name : str
        Name of the summary table.

    Returns
    -------
    List[str]
        Sorted list of unique hmm_domain_name values.
    """

    con = duckdb.connect(str(db_path), read_only=True)

    query = f"""
        SELECT DISTINCT hmm_domain_name
        FROM {table_name}
        ORDER BY hmm_domain_name
    """

    hmms = [row[0] for row in con.execute(query).fetchall()]

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


def extract_combined_summary(
    db: DuckDBClient, db_table_name: str, out_folder_path: Path, min_bit_score=0
):
    """
    TODO: Extract the combined hmm annotations across multiple organisms in the db.
    """
    raise NotImplementedError


# def extract_combined_summary(db_file_path: str,
#                                     db_table_name: str,
#                                     out_folder_path: str,
#                                     min_bit_score=0):
#     """
#     This is intended as a replacement for the combine step in the snakemake annotation pipelines
#     It is not an exact replacement because the outputs are dependedant on the values already in the database
#     This is a design decision. SQL operations and MSA logic need to keep the DB content in mind
#     """
#     this_file_dir = os.path.dirname(os.path.abspath(__file__))
#     logger.info(f'The table to query for hmm hits is {db_table_name}')
#     template_file_path = os.path.join(this_file_dir,'data_utils/sql_scripts/select_all_domain_n_genome.sql')
#     replacements = {
#         "TABLE_NAME": db_table_name,
#         "MIN_SCORE": f'{min_bit_score}'}
#     query = sql_utils.read_and_replace_query(template_file_path, replacements)
#     hhm_scan_tbl = sql_utils.run_sql_query(db_file_path, query)
#     counts_table_crosstab = pd.crosstab(hhm_scan_tbl['organism_name'], hhm_scan_tbl['hmm_domain_name'])

#     os.makedirs(out_folder_path, exist_ok=True)
#     out_hmm_fasta = os.path.join(out_folder_path, 'hmm_scan.csv')
#     counts_table_crosstab.to_csv(out_hmm_fasta)
