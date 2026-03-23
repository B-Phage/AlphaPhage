#!/usr/bin/env python3
"""
CLI wrapper to build the DuckDB summary HMM table from a data lake,
with a fingerprint file for DVC tracking.
"""

import argparse
import json
import pathlib
from pathlib import Path

import duckdb
import utils.consol_log as log
from hmm.duckdb_io import build_hmm_summary_table, fingerprint_hmm_table
from utils.config_loader import get_config
from utils.hash import hash_lake

logger = log.get_logger()

# --- Path Anchoring ---
SCRIPT_PATH = pathlib.Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parent.parent


def run_workflow(config_path, fingerprint_path=None, db_path_override=None):
    # 1. Load config
    hmm_cfg = get_config(override_path=config_path, params_set="hmm_config")

    # 2. Resolve Paths
    # Priority: CLI Override > Config Value
    db_path = Path(db_path_override) if db_path_override else Path(hmm_cfg.duckdb)
    # Ensure the parent directory for the DuckDB file exists
    db_path.parent.mkdir(parents=True, exist_ok=True)
    lake_root = Path(hmm_cfg.data_lake)
    table_name = hmm_cfg.summary_tbl

    # Fingerprint defaults to the config value if not provided by CLI
    f_path = Path(fingerprint_path) if fingerprint_path else Path(hmm_cfg.db_fingerprint)
    f_path.parent.mkdir(parents=True, exist_ok=True)

    # 3. Build summary table
    # This aggregates the parquet results from Step 2 into a DuckDB table
    build_hmm_summary_table(
        db_path=db_path,
        lake_root=lake_root,
        table_name=table_name,
        min_score=hmm_cfg.min_bit_score,
    )

    # 4. Fingerprinting (Critical for DVC tracking)
    con = duckdb.connect(str(db_path))
    try:
        table_hash = fingerprint_hmm_table(con, table_name)
        lake_hash = hash_lake(lake_root)
    finally:
        con.close()

    fingerprint_data = {
        "lake_hash": lake_hash,
        "table_hash": table_hash,
        "min_score": hmm_cfg.min_bit_score,
    }

    f_path.write_text(json.dumps(fingerprint_data, indent=2))

    logger.info(f"✅ HMM summary table built at: {db_path}")
    logger.info(f"📄 Fingerprint written to: {f_path}")


def main():
    parser = argparse.ArgumentParser(description="Build HMM summary DuckDB table.")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--fingerprint", help="Override path for DVC fingerprint file")
    parser.add_argument("--db_path", help="Override path for the DuckDB file")

    args = parser.parse_args()

    run_workflow(
        config_path=args.config, fingerprint_path=args.fingerprint, db_path_override=args.db_path
    )


if __name__ == "__main__":
    main()
