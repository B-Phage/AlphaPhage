import json
import subprocess
import sys
from pathlib import Path

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml
from hmm.duckdb_io import fingerprint_hmm_table, hash_file, hash_lake


# Helper: write a parquet file
def write_parquet(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df)
    pq.write_table(table, path)


# -------------------------------
# Integration test for full workflow
# -------------------------------
def test_hmm_summary_workflow(tmp_path: Path):
    """
    End-to-end integration test for the HMM summary workflow:
    - Creates fake parquet lake
    - Writes temporary YAML config
    - Runs main_cli
    - Verifies DuckDB summary table and fingerprint
    """
    # ---------------------------
    # Step 1: create parquet lake
    # ---------------------------
    lake_root = tmp_path / "lake"
    df1 = pd.DataFrame(
        {
            "protein_name": ["p1", "p2", "p3"],
            "organism_name": ["org1"] * 3,
            "hmm_domain_name": ["hmm1", "hmm2", "hmm3"],
            "annot": ["a1", "a2", "a3"],
            "evalue": [0.01, 0.02, 0.03],
            "score": [10, 5, 20],
            "bias": [0.1, 0.2, 0.3],
            "length": [50, 60, 70],
            "top_hmm_domain_hit_on_prot": [1, 0, 0],
            "top_protein_match_for_hmm": [1, 1, 1],
            "best_dom_env_start": [1, 1, 1],
            "best_dom_env_end": [50, 60, 70],
            "best_dom_env_length": [50, 60, 70],
            "sequence": ["AAAA", "BBBB", "CCCC"],
        }
    )

    # Create one organism folder
    org_path = lake_root / "organism=org1"
    parquet_file = org_path / "proteins.parquet"
    write_parquet(df1, parquet_file)

    # ---------------------------
    # Step 2: create YAML config
    # ---------------------------
    db_path = tmp_path / "summary.duckdb"
    fingerprint_path = tmp_path / "hmm_summary.fingerprint"

    config_data = {
        "hmm_config": {
            "duckdb": str(db_path),
            "data_lake": str(lake_root),
            "summary_tbl": "hmm_summary",
            "min_bit_score": 6,
        }
    }
    config_file = tmp_path / "config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config_data, f)

    # ---------------------------
    # Step 3: run CLI programmatically
    # ---------------------------
    # Import the CLI main function directly
    from workflows.hmm_parquet_to_summary_tbl import main_cli

    # Patch sys.argv
    sys_argv_backup = sys.argv
    sys.argv = [
        "hmm_parquet_to_summary_tbl.py",
        "--config",
        str(config_file),
        "--fingerprint",
        str(fingerprint_path),
    ]

    try:
        main_cli()
    finally:
        sys.argv = sys_argv_backup

    # ---------------------------
    # Step 4: check DuckDB table
    # ---------------------------
    con = duckdb.connect(str(db_path))
    rows = con.execute(
        "SELECT protein_name, hmm_domain_name FROM hmm_summary ORDER BY protein_name"
    ).fetchall()
    con.close()

    # Only rows passing filters:
    # top_protein_match_for_hmm=1 AND (top_hmm_domain_hit_on_prot=1 OR score>6)
    expected_rows = [("p1", "hmm1"), ("p3", "hmm3")]  # top_hmm_domain_hit_on_prot=1  # score=20>6
    assert rows == expected_rows

    # ---------------------------
    # Step 5: check fingerprint JSON
    # ---------------------------
    assert fingerprint_path.exists()
    fingerprint = json.loads(fingerprint_path.read_text())

    # fingerprint should contain the keys
    for key in ["lake_hash", "table_hash", "min_score"]:
        assert key in fingerprint

    # min_score matches config
    assert fingerprint["min_score"] == 6

    # table_hash and lake_hash are non-empty strings
    assert isinstance(fingerprint["table_hash"], str) and len(fingerprint["table_hash"]) == 64
    assert isinstance(fingerprint["lake_hash"], str) and len(fingerprint["lake_hash"]) == 64
