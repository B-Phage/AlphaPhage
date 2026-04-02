import argparse
import ast
from pathlib import Path
from typing import List

import pandas as pd
import utils.consol_log as log
from data_contracts.interaction_db import InteractionTableContract
from storage_io.duckdb_client import DuckDBClient
from utils.config_loader import get_config

logger = log.get_logger()


def extract_interaction_matrix(
    source_db_path: str,
    output_tsv: str,
    selected_sources: List[str],
    table_name: str | None = None,
):
    """Extracts interaction data from DuckDB and pivots it into a wide matrix.

    This function filters a long-format interaction table based on a list of
    trusted data sources, aggregates the maximum interaction value for each
    host-phage pair, and pivots the results into a wide 'Bacteria x Phage'
    matrix. Missing values are filled with the unknown constant defined in
    the InteractionTableContract.

    The resulting matrix is saved as a tab-separated values (TSV) file,
    where rows represent hosts (bacteria) and columns represent phages.

    Args:
        source_db_path: Path to the source DuckDB database file.
        output_tsv: Path where the resulting interaction matrix TSV will be saved.
        selected_sources: A list of strings representing the data sources
            (e.g., 'Laboratory_A', 'Public_DB_1') to include in the extraction.
        table_name: Optional override for the source table name. If None,
            the default TABLE_NAME from InteractionTableContract is used.

    Returns:
        None

    Raises:
        duckdb.IOException: If the source database cannot be accessed.
        pd.errors.EmptyDataError: If the query returns no data for the
            selected sources, preventing the pivot operation.
        OSError: If the output directory cannot be created or the file
            cannot be written.

    Example:
        >>> extract_interaction_matrix(
        ...     source_db_path="data/external.db",
        ...     output_tsv="data/matrix.tsv",
        ...     selected_sources=["Source_A", "Source_B"]
        ... )
        # Generates a TSV with hosts as index and phages as columns.
    """
    # Initialize the contract
    contract = InteractionTableContract()
    tbl = table_name or contract.TABLE_NAME

    # Ensure output directory exists
    Path(output_tsv).parent.mkdir(parents=True, exist_ok=True)

    with DuckDBClient(source_db_path, read_only=True) as client:
        # 1. Filter the long table for the sources you trust
        validate_schema(client, tbl, contract)
        source_filter = "', '".join(selected_sources)

        # Using contract attributes for column names
        query = f"""
            SELECT
                {contract.HOST},
                {contract.PHAGE},
                MAX({contract.INTERACTION}) as {contract.INTERACTION}
            FROM {tbl}
            WHERE {contract.SOURCE} IN ('{source_filter}')
            GROUP BY {contract.HOST}, {contract.PHAGE}
        """
        long_df = client.execute(query).df()

    if long_df.empty:
        logger.info(f"⚠️ Warning: No data found for sources: {selected_sources}")
        return

    # 2. Pivot to Wide Format using contract keys
    matrix_df = long_df.pivot(
        index=contract.HOST, columns=contract.PHAGE, values=contract.INTERACTION
    )

    # 3. Handle Missing Data using contract unknown value
    matrix_df = matrix_df.fillna(contract.VALUE_UNKNOWN).astype(int)

    # 4. Save to TSV
    matrix_df.to_csv(output_tsv, sep="\t")
    logger.info(f"✅ Interaction matrix saved to {output_tsv}")
    logger.info(f"📊 Dimensions: {matrix_df.shape[0]} hosts x {matrix_df.shape[1]} phages")


def _parse_list_input(input_val):
    """Safely converts string/list inputs to a Python list."""
    if not input_val:
        return []

    if isinstance(input_val, list):
        return input_val

    # If it's not a string, we can't parse or split it,
    # so just wrap it in a list and return.
    if not isinstance(input_val, str):
        return [input_val]

    try:
        parsed = ast.literal_eval(input_val)
        if isinstance(parsed, list):
            return parsed
    except (ValueError, SyntaxError):
        # Fallback for space-separated strings
        return input_val.split()

    # Final catch-all for strings that parsed into non-lists (e.g., "123")
    return [input_val]


def validate_schema(client: DuckDBClient, table_name: str, contract: InteractionTableContract):
    """Fails fast if the database table doesn't match the contract."""
    existing_columns = client.list_columns(table_name)
    for col in contract.required_columns:
        if col not in existing_columns:
            raise KeyError(
                f"❌ Schema Mismatch: Table '{table_name}' is missing "
                f"required column: '{col}'. Found: {existing_columns}"
            )


def main():
    parser = argparse.ArgumentParser(description="Extract interaction matrix from DuckDB.")
    parser.add_argument("-c", "--config", help="Path to params.yaml or override config")

    args = parser.parse_args()

    cfg = get_config(override_path=args.config, params_set="data_intake")

    extract_interaction_matrix(
        source_db_path=cfg.source_db,
        output_tsv=cfg.output_matrix,
        selected_sources=list(cfg.trusted_sources),
        table_name=getattr(cfg, "table_name", None),
    )


if __name__ == "__main__":
    main()
