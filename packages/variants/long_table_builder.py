import argparse
import json
from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd
import utils.consol_log as log
from storage_io.duckdb_client import DuckDBClient
from utils.config_loader import get_vocab

logger = log.get_logger()


def build_long_table(
    parquet_dir: str,
    db_path: str,
    table_name: str,
    vocab_section: str,
    config_path: Optional[str] = None,
    manifest_path: Optional[str] = None,
):
    """
    Aggregates Parquet shards into a DuckDB long table and applies token mapping.

    Args:
        parquet_dir: Path to the folder containing .parquet shards.
        db_path: Path to the persistent .db file.
        table_name: Name of the table to create (e.g., 'phage_variants').
        config_path: Path to the vocab YAML.
        vocab_section: The nested key in the YAML for this specific mapping.
    """
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    # 1. Connect to DuckDB (creates file if it doesn't exist)
    con = duckdb.connect(db_path)

    # 2. Extract the Vocab Mapping
    # Assuming get_vocab returns a dict where keys are variant_ids and values are tokens
    full_vocab = get_vocab(override_path=config_path)
    mapping_dict = full_vocab.get(vocab_section, {})
    clean_mapping = {str(k): v for k, v in mapping_dict.items()}

    # Convert dict to DataFrame for easy DuckDB ingestion
    mapping_df = pd.DataFrame(list(clean_mapping.items()), columns=["variant_id", "token_id"])
    # 3. Explicitly tell Pandas/DuckDB these are Strings and Integers
    mapping_df["variant_id"] = mapping_df["variant_id"].astype("string")
    mapping_df["token_id"] = pd.to_numeric(mapping_df["token_id"]).astype("int64")

    # 4. Use the Join with a Cast
    con.register("mapping_view", mapping_df)

    query = f"""
    CREATE OR REPLACE TABLE {table_name} AS
    SELECT
        p.organism_id,
        p.hmm_id,
        p.variant_id, -- Keep original type in the output if you want
        m.token_id,
        p.clustering_version
    FROM read_parquet('{parquet_dir}/*.parquet') p
    -- Force both sides to VARCHAR during the join to ensure '1' == 1
    LEFT JOIN mapping_view m
      ON CAST(p.variant_id AS VARCHAR) = CAST(m.variant_id AS VARCHAR)
    """
    con.execute(query)

    # 4. Cleanup and Summary
    # Execute and capture the result
    res = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()

    # Check if res exists before accessing index 0
    row_count = res[0] if res is not None else 0

    logger.info(f"Aggregated {row_count} rows into {table_name} at {db_path}")
    con.close()

    if manifest_path:
        with DuckDBClient(db_path) as client:
            table_hash = client.get_table_hash(table_name)

            # Write the "Receipt"
            manifest = {"table": table_name, "fingerprint": table_hash, "source": parquet_dir}
            Path(manifest_path).write_text(json.dumps(manifest, indent=2))


def cli():
    parser = argparse.ArgumentParser(
        description="Aggregate variant shards into a DuckDB long table."
    )
    parser.add_argument(
        "-i", "--parquet_dir", required=True, help="Input directory of parquet shards"
    )
    parser.add_argument("-d", "--db_path", required=True, help="Path to DuckDB file")
    parser.add_argument("-t", "--table_name", required=True, help="Target table name in DuckDB")
    parser.add_argument(
        "-s", "--vocab_section", required=True, help="Key in vocab config for token mapping"
    )
    parser.add_argument("-c", "--config_path", help="Optional override path for vocab config")
    parser.add_argument("-mp", "--manifest_path", help="Optional manifest path for table hash")

    args = parser.parse_args()

    build_long_table(
        parquet_dir=args.parquet_dir,
        db_path=args.db_path,
        table_name=args.table_name,
        vocab_section=args.vocab_section,
        config_path=args.config_path,
        manifest_path=args.manifest_path,
    )


if __name__ == "__main__":
    cli()
