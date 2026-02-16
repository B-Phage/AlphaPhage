#!/usr/bin/env python
"""
CLI wrapper to build the DuckDB summary HMM table from a data lake,
with a fingerprint file for DVC tracking.
"""

import argparse
import json
from pathlib import Path

import duckdb
import utils.consol_log as log
from hmm.duckdb_io import build_hmm_summary_table, fingerprint_hmm_table
from utils.config_loader import get_config
from utils.hash import hash_lake

logger = log.get_logger()


def main_cli():
    parser = argparse.ArgumentParser(
        description="Build HMM summary DuckDB table from parquet lake."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file containing hmm_config section",
    )
    parser.add_argument(
        "--code_hash",
        action="append",
        type=Path,
        required=True,
        help="Path to a hash file (repeatable)",
    )
    parser.add_argument(
        "--fingerprint",
        type=str,
        default="artifacts/hmm_summary.fingerprint",
        help="Path to write the fingerprint file for DVC tracking",
    )
    args = parser.parse_args()

    # Load config
    hmm_cfg = get_config(override_path=args.config, params_set="hmm_config")

    # Paths from config
    db_path = Path(hmm_cfg.duckdb)
    lake_root = Path(hmm_cfg.data_lake)
    table_name = hmm_cfg.summary_tbl

    # Ensure fingerprint folder exists
    fingerprint_out = Path(args.fingerprint)
    fingerprint_out.parent.mkdir(parents=True, exist_ok=True)

    # Use min_score from config
    min_score = hmm_cfg.min_bit_score

    # Build summary table
    build_hmm_summary_table(
        db_path=db_path,
        lake_root=lake_root,
        table_name=table_name,
        min_score=min_score,
    )

    con = duckdb.connect(str(db_path))
    table_hash = fingerprint_hmm_table(con, table_name)
    lake_hash = hash_lake(lake_root)
    con.close()

    fingerprint = {
        "lake_hash": lake_hash,
        "table_hash": table_hash,
        "min_score": min_score,
    }

    fingerprint_out.write_text(json.dumps(fingerprint, indent=2))

    logger.info(f"HMM summary table built at: {db_path}")
    logger.info(f"Fingerprint written to: {fingerprint_out}")


if __name__ == "__main__":
    main_cli()
