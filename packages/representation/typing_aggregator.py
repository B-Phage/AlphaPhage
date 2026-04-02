import argparse
import json
from pathlib import Path
from typing import Dict

import utils.consol_log as log
from storage_io.duckdb_client import DuckDBClient
from utils.config_loader import get_vocab

logger = log.get_logger()


def aggregate_serotypes(
    json_dir: str, db_path: str, table_name: str, manifest_path: str, vocab_path: str | None = None
):
    # 1. Load Vocab via internal utility
    vocab_cfg = get_vocab(override_path=vocab_path)

    if "vocab_best_match" not in vocab_cfg:
        raise KeyError("Section 'vocab_best_match' missing from loaded vocabulary.")

    # Map 'KL10': [0, 2] -> We keep the full list now
    # vocab_map = {'KL10': [0, 2], 'KL11': [0, 3], ...}
    vocab_map = {str(k): list(v) for k, v in vocab_cfg.vocab_best_match.items()}

    # Fallback for unknowns (usually [0, 0] or similar)
    unk_tokens = vocab_map.get("-", [0, 0])

    with DuckDBClient(db_path) as client:
        # 2. Extract raw data
        client.execute(
            f"""
            CREATE OR REPLACE TEMP TABLE raw_typing AS
            SELECT
                COALESCE(sample_name, regexp_extract(filename, '([^/]+)\\.json$', 1)) AS organism_id,
                best_match
            FROM read_json_auto('{json_dir}/*.json', filename=True)
        """
        )

        # 3. Build Multi-Column Token Mapping Query
        # We generate two CASE statements: one for Dim 1 and one for Dim 2
        cases_dim1 = "\n".join(
            [f"WHEN best_match = '{k}' THEN {v[0]}" for k, v in vocab_map.items() if k != "-"]
        )
        cases_dim2 = "\n".join(
            [f"WHEN best_match = '{k}' THEN {v[1]}" for k, v in vocab_map.items() if k != "-"]
        )

        query = f"""
        CREATE OR REPLACE TABLE {table_name} AS
        SELECT
            organism_id,
            best_match,
            CASE {cases_dim1} ELSE {unk_tokens[0]} END AS k_type_dim1,
            CASE {cases_dim2} ELSE {unk_tokens[1]} END AS k_type_dim2
        FROM raw_typing
        """
        client.execute(query)

        # 4. Cleanup
        client.execute(
            f"""
            UPDATE {table_name}
            SET k_type_dim1 = {unk_tokens[0]}, k_type_dim2 = {unk_tokens[1]}
            WHERE best_match IS NULL
        """
        )

        # 5. Manifest Generation for DVC tracking
        fingerprint = client.get_table_hash(table_name)
        manifest = {
            "table": table_name,
            "fingerprint": fingerprint,
            "vocab_source": vocab_path or "params.yaml default",
            "type": "serotype_token_aggregation",
        }

        m_path = Path(manifest_path)
        m_path.parent.mkdir(parents=True, exist_ok=True)
        m_path.write_text(json.dumps(manifest, indent=2))

        logger.info(f"✅ Serotype tokens aggregated into {table_name}. Fingerprint: {fingerprint}")


def main():
    parser = argparse.ArgumentParser(description="Aggregate host serotypes into DuckDB tokens.")
    parser.add_argument(
        "-i", "--json_dir", required=True, help="Directory containing Kaptive .json files"
    )
    parser.add_argument("-d", "--db_path", required=True, help="Path to the features.duckdb")
    parser.add_argument(
        "-t", "--table_name", required=True, help="Target table name (e.g. bacteria_serotypes)"
    )
    parser.add_argument("-m", "--manifest_path", required=True, help="DVC output manifest (.json)")
    parser.add_argument("-v", "--vocab_path", help="Optional override path to vocab.yaml")

    args = parser.parse_args()

    aggregate_serotypes(
        json_dir=args.json_dir,
        db_path=args.db_path,
        table_name=args.table_name,
        manifest_path=args.manifest_path,
        vocab_path=args.vocab_path,
    )


if __name__ == "__main__":
    main()
