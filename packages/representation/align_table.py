from pathlib import Path
from typing import List, Optional

import pandas as pd
import utils.consol_log as log
from omegaconf import DictConfig

logger = log.get_logger()

# ===============================
# Core functions for table/msa processing
# ===============================


def read_table(in_file: str, file_format: str) -> pd.DataFrame:
    """
    Reads a table or MSA file and returns a DataFrame.

    Supported formats:
      - 'csv' : comma-separated table with header, index in first column
      - 'tsv' : tab-separated table with header, index in first column
      - 'csa' : custom aligned sequence format (first col = sequence, second col = name)
    """
    if file_format == "csv":
        df = pd.read_csv(in_file, sep=",", index_col=0)
    elif file_format == "tsv":
        df = pd.read_csv(in_file, sep="\t", index_col=0)
    elif file_format == "csa":
        df1 = pd.read_csv(in_file, sep=",", header=None)
        sequences = df1[0].values
        indices = df1[1].values
        df = pd.DataFrame([list(seq) for seq in sequences], index=indices)
    else:
        raise ValueError(f"Unrecognized file_format '{file_format}'. Exiting.")
    df.columns = [str(c) for c in df.columns]
    return df


def split_index_column(df: pd.DataFrame, sep: str, index_element: int) -> pd.DataFrame:
    """
    Splits multi-part index strings and keeps only the specified element.

    Example:
      index = "genomeA;proteinX"
      sep=";", index_element=0 -> "genomeA"
    """
    new_index = [str(idx).split(sep)[index_element] for idx in df.index]
    df.index = new_index
    return df


def _align_rows(df: pd.DataFrame, row_names: list, filler_char: str) -> pd.DataFrame:
    """
    Aligns the DataFrame rows to a reference list of row_names.
    Missing rows are filled with filler_char from the vocab.
    """
    df_aligned = []
    columns = df.columns
    for name in row_names:
        if name in df.index:
            df_aligned.append(df.loc[name])
        else:
            filler_series = pd.Series([filler_char] * len(columns), index=columns, name=name)
            df_aligned.append(filler_series)
    df_out = pd.DataFrame(df_aligned)
    df_out.columns = columns
    return df_out


def select_columns(df: pd.DataFrame, col_names: list) -> pd.DataFrame:
    """
    Selects only the desired columns from the DataFrame.

    Rules:
      - col_names=None or [None] -> keep all columns
      - col_names contains more than one None -> error
    """
    if col_names is None or col_names == [None]:
        return df
    if None in col_names:
        logger.error(f"Invalid col_names: {col_names}. Only one None allowed. Exiting.")
        raise ValueError("col_names cannot contain None.")
    if any("\n" in str(c) for c in col_names):
        logger.error('"\\n" found in col_names. Exiting.')
        raise ValueError("col_names cannot contain newline characters.")
    df_out = df[col_names]
    return df_out


# ===============================
# High-level wrapper
# ===============================


def process_table(
    in_file: str,
    file_format: str,
    vocab: DictConfig,
    row_names: Optional[List[str]] = None,
    col_names: Optional[List[str]] = None,
    sep_index_split: str = ";",
    index_element: int | None = None,
) -> pd.DataFrame:
    """
    Reads and processes a table or MSA file into a pandas DataFrame,
    optionally aligning rows and selecting columns using a vocab-defined filler character.

    This function replicates the functionality of `order_select` in a modular way,
    providing a consistent pipeline for importing, aligning, and filtering tabular data.
    It supports CSV, TSV, and custom MSA formats (e.g., `.csa`).

    Processing steps:
        1. Reads the input file into a pandas DataFrame.
        2. Optionally splits the index column using `sep_index_split` and selects a specific element.
        3. Optionally aligns rows to a reference list (`row_names`), filling missing rows with `vocab.filler_char`.
        4. Optionally selects a subset of columns (`col_names`).

    Args:
        in_file (str): Path to the input file (CSV, TSV, or custom MSA).
        file_format (str): Format of the input file. Supported values: `'csv'`, `'tsv'`, `'csa'`.
        vocab (DictConfig): Vocabulary configuration containing at minimum a `filler_char`
            used to fill missing rows.
        row_names (Optional[List[str]]): Reference list of row names to align the DataFrame to.
            Rows not present in the file are filled using `vocab.filler_char`. Defaults to None.
        col_names (Optional[List[str]]): List of column names to select. If None, all columns are retained.
            Defaults to None.
        sep_index_split (str, optional): Separator to use when splitting the index column. Defaults to ';'.
        index_element (int | None, optional): Element of the split index to use as the new row index.
            If None, the index is unchanged. Defaults to None.

    Returns:
        pd.DataFrame: Processed DataFrame with rows optionally aligned and columns optionally filtered.
            - Index: original or split column values.
            - Columns: selected columns or all columns if `col_names` is None.

    Raises:
        ValueError: If the file format is not recognized or internal checks fail (e.g., col_names invalid).
    """
    df = read_table(in_file, file_format)
    if index_element is not None:
        df = split_index_column(df, sep_index_split, index_element)
    if row_names is not None:
        df = _align_rows(df, row_names, vocab.filler_char)
    if col_names is not None:
        df = select_columns(df, col_names)
    return df


def process_msa_table(
    msa_file: str | Path,
    vocab: DictConfig,
    row_names: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Reads and processes a custom MSA (.csa) file into a pandas DataFrame,
    optionally aligning rows to a reference list using a vocab-defined filler character.

    This function is a specialized version of `process_table` for multiple sequence
    alignments, designed to simplify preprocessing for variant clustering or feature
    engineering pipelines.

    The `.csa` file format is expected as:

        <alignment_entry><organism_name>;<protein_name>

    Example line:

        ------------------------L----YIRNNGYYCYR--DprTGKE--...--grewdkieik*...,411_P2;411_P2_33

    - `alignment_entry`: Aligned protein sequence, with gap characters or placeholders.
    - `organism_name`: Source organism of the protein.
    - `protein_name`: Protein identifier.

    Processing steps:
        1. Reads the `.csa` file as a table with sequences split into single-character columns.
        2. Splits the index column using ';' and keeps the first element (organism name) as the row index.
        3. If `row_names` is provided, aligns the DataFrame rows to this list, filling missing rows
           with the filler character defined in `vocab.filler_char`.

    Args:
        msa_file (str | Path): Path to the `.csa` MSA file.
        vocab (DictConfig): Vocabulary configuration containing at minimum a `filler_char`
            used to fill missing rows.
        row_names (Optional[List[str]]): Reference list of row names to align the DataFrame to.
            Rows not present in the file will be filled using `vocab.filler_char`. Defaults to None.

    Returns:
        pd.DataFrame: DataFrame where each row corresponds to an MSA entry (protein),
                      columns represent alignment positions as single-character strings,
                      and the index is the organism name or other selected element from
                      the split index.

    Raises:
        ValueError: If the file does not have a `.csa` extension.
    """
    sep_index_split = ";"  #';' in 'genome;protein'
    index_element = 0  # e.g. genome

    msa_file = Path(msa_file)
    msa_file_format = msa_file.suffix.lstrip(".")  # removes leading dot

    if msa_file_format != "csa":
        raise ValueError(f"Expected a .csa MSA file, got {msa_file_format}")

    df = read_table(str(msa_file), msa_file_format)
    df = split_index_column(df, sep_index_split, index_element)
    if row_names is not None:
        df = _align_rows(df, row_names, vocab.filler_char)

    return df
