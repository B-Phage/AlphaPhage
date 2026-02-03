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
    rows = [
        ("prot1", "org1", "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ", "hmmA"),
        ("prot2", "org2", "GAVLIPFSTWYQNKMDHEKR", "hmmA"),
        ("prot3", "org1", "MKQLLFLVLLFSSAYSRG", "hmmB"),
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
        "hmm_config": {
            "duckdb": str(db_path),
            "summary_tbl": table_name,
            "hmm_fasta_dir": str(fasta_dir),
        }
    }
    cfg_path = tmp_path / "hmm_config.yaml"
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f)
    return cfg_path, fasta_dir


def test_create_hmm_fastas_cli(tmp_config):
    """Integration test: call workflows.create_hmm_fastas.py as a CLI."""
    cfg_path, fasta_dir = tmp_config

    # Run the script via subprocess
    cmd = [
        sys.executable,  # ensures we use the same Python env
        "workflows/create_hmm_fastas.py",
        "--config",
        str(cfg_path),
        "--force",
    ]
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,  # decode bytes to str
        check=False,  # we’ll handle assertion ourselves
    )

    print("STDOUT:\n", result.stdout)
    print("STDERR:\n", result.stderr)

    # Now assert exit code
    assert result.returncode == 0, f"CLI failed: {result.stderr}"

    # Verify FASTA files are created
    fasta_files = list(fasta_dir.glob("*.fasta"))
    assert len(fasta_files) == 2  # hmmA and hmmB

    # Check content of one file
    hmmA_fasta = fasta_dir / "hmmA.fasta"
    content = hmmA_fasta.read_text()
    assert ">org1;prot1" in content
    assert ">org2;prot2" in content
    assert "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ" in content
    assert "GAVLIPFSTWYQNKMDHEKR" in content
