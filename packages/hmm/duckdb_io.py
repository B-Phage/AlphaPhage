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

import math
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import utils.consol_log as log
from hmm.search import normalize_name
from hmm.types import Protein
from utils.duckdb_client import DuckDBClient

logger = log.get_logger(level=log.DEBUG)


def assert_duckdb_safe(df: pd.DataFrame, numeric_cols: list[str] | None = None) -> None:
    """
    Assert that a DataFrame is safe for DuckDB insertion:
      - Object columns contain only str or None
      - Numeric columns (explicitly listed) contain only int, float, or None
    """
    numeric_cols = numeric_cols or []

    for col in df.columns:
        series = df[col]
        if col in numeric_cols:
            for i, val in enumerate(series):
                if val is not None and not isinstance(val, (int, float)):
                    raise TypeError(f"Column {col} contains non-numeric values after conversion!")
        elif series.dtype == "O":
            if not series.map(lambda x: isinstance(x, str) or x is None).all():
                raise TypeError(f"Column {col} contains non-str values after conversion!")


def duckdb_force_string(x: Any) -> str | None:
    """
    Force a value to a Python string, returning None for missing/NA.
    """
    if x is None or pd.isna(x) or (isinstance(x, float) and math.isnan(x)):
        return None
    if isinstance(x, bytes):
        return x.decode("utf-8")
    if isinstance(x, bytearray):
        return bytes(x).decode("utf-8")
    if isinstance(x, memoryview):
        return x.tobytes().decode("utf-8")
    return str(x)


def duckdb_safe_numeric(series: pd.Series) -> pd.Series:
    """
    Convert a pandas Series to a numeric type safe for DuckDB:
      - Keep ints/floats
      - Convert invalid entries to None
    """
    # coerce invalid entries to NaN
    numeric_series = pd.to_numeric(series, errors="coerce")
    # replace NaN with None for DuckDB
    return numeric_series  # keep dtype numeric


def duckdb_prepare_dataframe(
    df: pd.DataFrame, string_cols: list[str] | None = None, num_cols: list[str] | None = None
) -> pd.DataFrame:
    """
    Make a dataframe fully safe for DuckDB insertion.

    Parameters
    ----------
    df : pd.DataFrame
        Original dataframe
    string_cols : list[str] | None
        Columns to forcibly convert to strings.
    num_cols : list[str] | None
        Columns to forcibly convert to numbers.

    Returns
    -------
    pd.DataFrame
        A copy of the dataframe safe for DuckDB
    """
    df_safe = df.copy()

    # # force string columns first
    if string_cols:
        for col in string_cols:
            df_safe[col] = df_safe[col].map(duckdb_force_string).astype("string")

    # Then fix numeric columns
    if num_cols:
        for col in num_cols:
            df_safe[col] = duckdb_safe_numeric(df_safe[col])

    return df_safe


def write_hmm_results_with_sequences(
    hmm_scan_df: pd.DataFrame,
    protein_py_easel_obj: Iterable[Protein],
    db: DuckDBClient,
    sql_table_name: str,
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

    db_path : str
        Path to the DuckDB database file.

    sql_table_name : str
        Name of the DuckDB table to create or append to.

    Returns
    -------
    None

    Side Effects
    ------------
    - Creates a DuckDB table if it does not already exist.
    - Inserts or upserts rows into the specified table.

    Notes
    -----
    This function performs no validation or filtering of HMM results and
    assumes all upstream HMM analysis has already been completed. It is
    intended as an I/O boundary between HMM scanning and downstream workflows
    such as multiple sequence alignment (MSA).
    """
    if hmm_scan_df.empty:
        # No hits: just exit
        logger.warning(f"No HMM hits to insert")
        return

    col_from_hmm_scan = [
        "protein_name",
        "organism_name",
        "hmm_domain_name",
        "annot",
        "evalue",
        "score",
        "bias",
        "length",
        "top_hmm_domain_hit_on_prot",
        "top_protein_match_for_hmm",
        "best_dom_env_start",
        "best_dom_env_end",
        "best_dom_env_length",
    ]
    # Plus columns from protein file; ['sequence']

    if not db.table_exists(sql_table_name):
        logger.warning(f"Creating {sql_table_name}")
        db.con.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {sql_table_name} (
                protein_name VARCHAR,
                organism_name VARCHAR,
                hmm_domain_name VARCHAR,
                annot VARCHAR,
                evalue FLOAT,
                score FLOAT,
                bias FLOAT,
                length INTEGER,
                top_hmm_domain_hit_on_prot INTEGER,
                top_protein_match_for_hmm INTEGER,
                best_dom_env_start INTEGER,
                best_dom_env_end INTEGER,
                best_dom_env_length INTEGER,
                sequence VARCHAR,
                PRIMARY KEY (protein_name, organism_name, hmm_domain_name)
            )
            """
        )
    df_for_sql = hmm_scan_df[col_from_hmm_scan]

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
    df_for_sql = pd.merge(df_for_sql, seq_df, how="outer", on="protein_name")

    # Force *pure Python object strings*, not pandas StringDtype
    # and normalise object-dtype columns to avoid DuckDB / pandas incompatibilities
    STRING_COLUMNS = ["protein_name", "organism_name", "hmm_domain_name", "annot", "sequence"]
    NUMERIC_COLUMNS = [
        "evalue",
        "score",
        "bias",
        "length",
        "top_hmm_domain_hit_on_prot",
        "top_protein_match_for_hmm",
        "best_dom_env_start",
        "best_dom_env_end",
        "best_dom_env_length",
    ]
    df_for_sql = duckdb_prepare_dataframe(
        df_for_sql, string_cols=STRING_COLUMNS, num_cols=NUMERIC_COLUMNS
    )
    assert_duckdb_safe(df_for_sql, numeric_cols=NUMERIC_COLUMNS)

    db.insert_dataframe(
        sql_table_name,
        df_for_sql,
        primary_keys=["protein_name", "organism_name", "hmm_domain_name"],
    )


def read_hmm_list(
    db: DuckDBClient,
    db_table_name: str,
    min_bit_score: float = 0,  # the score is always > 0, so default to no score filter
):
    """
    TODO: Get the list of unique/distinct HMM annotations in the table where the min_bit_score is
    greater than the given value or the HMM is the best hit for a given protein.
    """
    raise NotImplementedError


def extract_hmm_sequences_for_msa(
    hmm_annotation: str,
    db: DuckDBClient,
    db_table_name: str,
    out_folder_path: str,
    min_bit_score=0,  # the score is always > 0, so default to no score filter
    out_file_extension: str = "fa",
):
    """
    TODO: Extract sequences matching a specific HMM for MSA generation.
    """
    raise NotImplementedError


def extract_combined_summary(
    db: DuckDBClient, db_table_name: str, out_folder_path: Path, min_bit_score=0
):
    """
    TODO: Extract the combined hmm annotations across multiple organisms in the db.
    """
    raise NotImplementedError


# def read_hmm_list(db_table_name,
#                  db_file_path,
#                  min_bit_score=0):
#     """
#     Get the list of unique/distinct HMM annotations in the table where the min_bit_score is
#     greater than the given value or the HMM is the best hit for a given protein.
#     """
#     this_file_dir = os.path.dirname(os.path.abspath(__file__))
#     logger.info(f'The table to query for hmm hits is {db_table_name}')
#     template_file_path = os.path.join(this_file_dir, 'data_utils/sql_scripts/select_unique_hmms.sql')
#     replacements = {
#         "TABLE_NAME": db_table_name,
#         "MIN_SCORE": f'{min_bit_score}'}
#     query = sql_utils.read_and_replace_query(template_file_path, replacements)
#     hmm_tbl = sql_utils.run_sql_query(db_file_path, query)

#     logger.info(f'The hmm query results are {hmm_tbl}')
#     logger.info(f'hmm result type: {type(hmm_tbl)}')
#     hmm_list = hmm_tbl['hmm_domain_name'].to_list()

#     return hmm_list


# def extract_hmm_sequences_for_msa(hmm_annotation: str,
#                                            db_file_path: str,
#                                            db_table_name: str,
#                                            out_folder_path: str,
#                                            min_bit_score=0, # the score is always > 0, so default to no score filter
#                                            out_file_extension: str='fa'):
#     """
#     Get all hits of a particular HMM, eg "phrog_1"
#     This step is required to get the data for an MSA

#     If you want a subset of organisms. The optimal thing to do is to make a new table
#     i.e. an inner join of a table with a organisms lookup and the sequence hmm hits table
#     then use that table in this query (db_table_name)
#     """

#     this_file_dir = os.path.dirname(os.path.abspath(__file__))
#     logger.info(f'The table to query for hmm hits is {db_table_name}')
#     template_file_path = os.path.join(this_file_dir,'data_utils/sql_scripts/select_hmm_sequences.sql')
#     replacements = {
#         "TABLE_NAME": db_table_name,
#         "HMM_ANNOT": hmm_annotation,
#         "MIN_SCORE": f'{min_bit_score}'}
#     query = sql_utils.read_and_replace_query(template_file_path, replacements)
#     hhm_seq_tbl = sql_utils.run_sql_query(db_file_path, query)

#     os.makedirs(out_folder_path, exist_ok=True)
#     out_hmm_fasta = os.path.join(out_folder_path, f'{hmm_annotation}.{out_file_extension}')

#     # write the sql output to a fasta file
#     with open(out_hmm_fasta, 'w') as hmm_fasta_file:
#         for _, row in hhm_seq_tbl.iterrows():
#             query_genome = row['organism_name']
#             query_prot = row['protein_name']
#             sequence = row['sequence']
#             hmm_fasta_file.write(f'>{query_genome};{query_prot}\n')
#             hmm_fasta_file.write(f'{sequence}\n')


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
