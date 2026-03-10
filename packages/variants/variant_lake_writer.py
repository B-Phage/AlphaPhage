# variant_lake_writer.py
from pathlib import Path

import pandas as pd
from data_contracts.variant_lake import VARIANT_LAKE_PRIMARY_KEYS, VARIANT_LAKE_SCHEMA
from storage_io.parquet_lake import enforce_schema, write_parquet_partition


def write_variant_shard(df: pd.DataFrame, out_path: Path) -> None:
    """
    Domain-specific writer for variant data.
    Uses generic storage_io utilities to enforce the variant data contract.
    """
    # 1. Domain Logic: Ensure IDs are present
    if "organism_id" not in df.columns or "hmm_id" not in df.columns:
        raise ValueError("Variant dataframe must contain 'organism_id' and 'hmm_id'")

    # 2. Infrastructure: Enforce the schema defined in data_contracts
    df = enforce_schema(df, VARIANT_LAKE_SCHEMA)

    # 3. Infrastructure: Write the immutable partition
    write_parquet_partition(df=df, out_path=out_path, key_cols=VARIANT_LAKE_PRIMARY_KEYS)
