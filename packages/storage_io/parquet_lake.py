from pathlib import Path

import pandas as pd


def write_parquet_partition(df: pd.DataFrame, out_path: Path, key_cols: list) -> None:
    """
    Write a pandas DataFrame to a Parquet file as an immutable partition.

    This function writes a DataFrame to the specified Parquet file path,
    creating parent directories as needed. It enforces an immutable, single-
    file write, which ensures that each partition in a Parquet lake is
    deterministic and safely versioned with DVC. Use this function instead
    of direct `to_parquet` calls when managing per-partition outputs in a
    data lake to maintain schema consistency and reproducibility.

    Args:
        df (pd.DataFrame): The DataFrame to write. Should already be aligned
            to the expected schema for this partition.
        out_path (Path): The full path where the Parquet file will be written.
            Parent directories will be created automatically if they do not exist.

    Returns:
        None

    Notes:
        - The write is **overwrite-only**; existing files at `out_path` will
          be replaced. Do not append to existing partitions in-place.
        - The DataFrame index is not written; only columns in the DataFrame
          are persisted.
    """
    if df.empty:
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)

    duplicates = df.duplicated(subset=key_cols)
    if duplicates.any():
        raise ValueError(f"Duplicate keys detected in partition")

    df.to_parquet(
        out_path,
        engine="pyarrow",
        index=False,
    )


def enforce_schema(df: pd.DataFrame, schema: dict[str, str]) -> pd.DataFrame:
    """
    Align a DataFrame to a specified schema before writing to a Parquet lake.

    This function ensures that a DataFrame conforms to the expected column
    names and data types for a Parquet partition. It adds missing columns
    with null values, casts existing columns to the desired types, and drops
    any extra columns. Use this function to prevent schema drift across
    partitions and to maintain compatibility for downstream queries and
    analytics.

    Args:
        df (pd.DataFrame): The input DataFrame to be aligned.
        schema (dict[str, str]): A mapping from column names to expected data
            types (e.g., {"protein_name": "string", "length": "int"}).

    Returns:
        pd.DataFrame: A new DataFrame with columns and data types aligned to
        the provided schema, ready for safe Parquet writing.
    """
    # Add missing columns with nulls
    for col, dtype in schema.items():
        if col not in df.columns:
            df[col] = pd.NA
    # Cast columns to expected types
    for col, dtype in schema.items():
        if dtype.startswith("int"):
            df[col] = df[col].astype("Int64")  # nullable integer
        elif dtype.startswith("float"):
            df[col] = df[col].astype("float")
        elif dtype == "string":
            df[col] = df[col].astype("string")
    # Drop extra columns
    df = df[list(schema.keys())]
    return df


"""
Other useful utilities:


def get_parquet_schema(path_glob) -> pa.Schema
def count_files(path_glob) -> int
def list_columns(path_glob) -> list[str]
def organism_partition_path(root: Path, organism: str) -> Path
def list_partitions(root: Path) -> dict[str, list[str]]

"""
