import subprocess
import sys
from pathlib import Path

import duckdb
import pytest
import yaml


@pytest.fixture
def tmp_db_and_table(tmp_path):
    """Create a temporary DuckDB database with a summary table and dummy data."""
    db_path = tmp_path / "test.duckdb"
    table_name = "hmm_summary"

    con = duckdb.connect(str(db_path))
    con.execute(
        f"""
        CREATE TABLE {table_name} (
            protein_name TEXT,
            organism_name TEXT,
            sequence TEXT,
            hmm_domain_name TEXT
        )
    """
    )
    # Updated rows: hmmA has 2 copies, hmmB now has 2 copies to meet the min_copies=2 threshold.
    rows = [
        ("prot1", "org1", "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ", "hmmA"),
        ("prot2", "org2", "GAVLIPFSTWYQNKMDHEKR", "hmmA"),
        ("prot3", "org1", "MKQLLFLVLLFSSAYSRG", "hmmB"),
        ("prot4", "org3", "MKQLLFLVLLFSSAYSRG", "hmmB"),  # Added this row
    ]
    con.executemany(f"INSERT INTO {table_name} VALUES (?, ?, ?, ?)", rows)
    con.close()
    return db_path, table_name


@pytest.fixture
def tmp_config(tmp_path, tmp_db_and_table):
    """Create a temporary YAML config file pointing to the temp DuckDB and FASTA output dir."""
    db_path, table_name = tmp_db_and_table
    fasta_dir = tmp_path / "fasta"
    fasta_dir.mkdir()
    cfg = {
        "general": {"name": "test", "code_version": "v0.0.1"},
        "hmm_config": {
            "duckdb": str(db_path),
            "summary_tbl": table_name,
            "hmm_fasta_dir": str(fasta_dir),
        },
    }
    cfg_path = tmp_path / "hmm_config.yaml"
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f)
    return cfg_path, fasta_dir


def test_create_hmm_fastas_cli(tmp_config):
    """Integration test: call workflows.create_hmm_fastas.py as a CLI."""
    cfg_path, fasta_dir = tmp_config

    cmd = [
        sys.executable,
        "workflows/create_hmm_fastas.py",
        "--config",
        str(cfg_path),
        "--force",
    ]
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )

    assert result.returncode == 0, f"CLI failed: {result.stderr}"

    # Verify FASTA files are created
    fasta_files = list(fasta_dir.glob("*.fasta"))

    # Logic Check:
    # hmmA has 2 entries -> Created
    # hmmB has 2 entries -> Created
    assert len(fasta_files) == 2

    # Check content of hmmA
    hmmA_fasta = fasta_dir / "hmmA.fasta"
    content_a = hmmA_fasta.read_text()
    assert ">org1;prot1" in content_a
    assert ">org2;prot2" in content_a

    # Check content of hmmB to ensure the new row is there
    hmmB_fasta = fasta_dir / "hmmB.fasta"
    content_b = hmmB_fasta.read_text()
    assert ">org1;prot3" in content_b
    assert ">org3;prot4" in content_b
