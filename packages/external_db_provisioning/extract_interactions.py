import argparse
import ast
from pathlib import Path
from typing import List

import pandas as pd
import utils.consol_log as log
from storage_io.duckdb_client import DuckDBClient
from utils.config_loader import get_config

logger = log.get_logger()


def extract_interaction_matrix(
    source_db_path: str,
    output_tsv: str,
    selected_sources: List[str],
    table_name: str = "interaction_table",
):
    """
    Extracts specific sources from the RnD DB and pivots them into a
    wide matrix (Bacteria x Phage).
    """
    # Ensure output directory exists
    Path(output_tsv).parent.mkdir(parents=True, exist_ok=True)

    with DuckDBClient(source_db_path, read_only=True) as client:
        # 1. Filter the long table for the sources you trust
        # We use tuple formatting to safely handle the SQL IN clause
        source_filter = "', '".join(selected_sources)

        query = f"""
            SELECT host, phage, MAX(interaction) as interaction
            FROM {table_name}
            WHERE source IN ('{source_filter}')
            GROUP BY host, phage
        """
        long_df = client.execute(query).df()

    if long_df.empty:
        logger.info(f"⚠️ Warning: No data found for sources: {selected_sources}")
        return

    # 2. Pivot to Wide Format (Index = Bacteria/host, Columns = Phage)
    matrix_df = long_df.pivot(index="host", columns="phage", values="interaction")

    # 3. Handle Missing Data (-1 represents 'Not Tested')
    matrix_df = matrix_df.fillna(-1).astype(int)

    # 4. Save to TSV
    matrix_df.to_csv(output_tsv, sep="\t")
    logger.info(f"✅ Interaction matrix saved to {output_tsv}")
    logger.info(f"📊 Dimensions: {matrix_df.shape[0]} hosts x {matrix_df.shape[1]} phages")


def _parse_list_input(input_val):
    """
    Safely converts a stringified list or a space-separated string
    into a proper Python list.
    """
    if not input_val:
        return []

    # If it's already a list (from some internal calls), return it
    if isinstance(input_val, list):
        return input_val

    try:
        # This handles "['Source_A', 'Source_B']" -> ['Source_A', 'Source_B']
        parsed = ast.literal_eval(input_val)
        if isinstance(parsed, list):
            return parsed
    except (ValueError, SyntaxError):
        # Fallback: Treat as a space-separated string if literal_eval fails
        return input_val.split()

    return [input_val]


def main():
    parser = argparse.ArgumentParser(description="Extract interaction matrix from DuckDB.")
    parser.add_argument("-c", "--config", help="Path to params.yaml or override config")

    args = parser.parse_args()

    # Load the structured config using your function
    cfg = get_config(override_path=args.config, params_set="data_intake")

    # Use values from config, allowing CLI override for output path if needed
    extract_interaction_matrix(
        source_db_path=cfg.source_db,
        output_tsv=cfg.output_matrix,
        selected_sources=list(cfg.trusted_sources),
        table_name=cfg.table_name,
    )


if __name__ == "__main__":
    main()
