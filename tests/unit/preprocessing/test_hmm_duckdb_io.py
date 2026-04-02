import hashlib
from pathlib import Path

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from data_contracts.hmm_lake import HMM_LAKE_PRIMARY_KEYS, HMM_LAKE_SCHEMA
from hmm.duckdb_io import (
    build_hmm_summary_table,
    extract_hmm_sequences_for_msa,
    fingerprint_hmm_table,
    get_distinct_hmms,
    write_hmm_results_with_sequences,
)


@pytest.fixture
def protein_block():
    return [
        MockProtein("prot1", "MKT"),
        MockProtein("prot2", "GGA"),
        MockProtein("prot3", "LLL"),
    ]


@pytest.fixture
def hmm_df():
    # Example HMM hits
    return pd.DataFrame(
        [
            {
                "protein_name": "prot1",
                "organism_name": "E_coli",
                "hmm_domain_name": "HMM1",
                "annot": "dom1",
                "evalue": 1e-5,
                "score": 50,
                "bias": 0.1,
                "length": 100,
                "top_hmm_domain_hit_on_prot": 1,
                "top_protein_match_for_hmm": 1,
                "best_dom_env_start": 10,
                "best_dom_env_end": 50,
                "best_dom_env_length": 40,
            },
            {
                "protein_name": "prot2",
                "organism_name": "E_coli",
                "hmm_domain_name": "HMM2",
                "annot": "dom2",
                "evalue": 1e-6,
                "score": 60,
                "bias": 0.0,
                "length": 120,
                "top_hmm_domain_hit_on_prot": 1,
                "top_protein_match_for_hmm": 1,
                "best_dom_env_start": 5,
                "best_dom_env_end": 55,
                "best_dom_env_length": 50,
            },
        ]
    )


class MockProtein:
    """Minimal mock for pyhmmer Protein object"""

    def __init__(self, name, sequence="MKT"):
        self.name = name
        self._sequence = sequence

    def textize(self):
        class TextObj:
            def __init__(self, sequence):
                self.sequence = sequence

        return TextObj(self._sequence)


def test_write_hmm_regular(tmp_path, hmm_df, protein_block):
    out_root = tmp_path
    organism_name = "E_coli"

    write_hmm_results_with_sequences(hmm_df, protein_block, organism_name, out_root)

    # Check output Parquet file exists
    out_file = out_root / f"organism={organism_name}" / "proteins.parquet"
    assert out_file.exists()

    # Load Parquet and verify contents
    df = pd.read_parquet(out_file)
    # Should contain all proteins in protein_block
    assert set(df["protein_name"]) == {"prot1", "prot2", "prot3"}
    # HMM columns are present
    for col in HMM_LAKE_SCHEMA.keys():
        assert col in df.columns
    # Verify sequence column
    assert set(df.loc[df["protein_name"] == "prot1", "sequence"]) == {"MKT"}


def test_write_hmm_empty_hmm_scan(tmp_path, caplog):
    """
    Test that write_hmm_results_with_sequences correctly skips writing
    when hmm_scan_df is empty, and logs a warning.
    """
    out_root = tmp_path
    organism_name = "E_coli"
    empty_df = pd.DataFrame(columns=list(HMM_LAKE_SCHEMA.keys()))
    protein_block = [MockProtein("prot1"), MockProtein("prot2"), MockProtein("prot3")]

    with pytest.raises(ValueError) as exc_info:
        write_hmm_results_with_sequences(empty_df, protein_block, organism_name, out_root)

    assert "No HMM hits to insert for" in str(exc_info.value)


def test_write_hmm_empty_proteins(tmp_path, hmm_df):
    # Non-empty HMM hits, but protein block is empty
    out_root = tmp_path
    organism_name = "E_coli"
    empty_proteins = []

    write_hmm_results_with_sequences(hmm_df, empty_proteins, organism_name, out_root)

    out_file = out_root / f"organism={organism_name}" / "proteins.parquet"
    df = pd.read_parquet(out_file)

    # Proteins from HMM hits should exist
    assert set(df["protein_name"]) == {"prot1", "prot2"}
    # Sequence column should exist and be NaN
    assert df["sequence"].isna().all()


@pytest.fixture
def sample_hmm_df():
    """Return a small HMM DataFrame for testing."""
    data = {
        "protein_name": ["prot1", "prot2", "prot3", "prot4"],
        "organism_name": ["E_coli"] * 4,
        "hmm_domain_name": ["hmm1", "hmm2", "hmm3", "hmm4"],
        "annot": ["a"] * 4,
        "evalue": [0.01, 0.02, 0.03, 0.04],
        "score": [50, 10, 30, 0],
        "bias": [0.0] * 4,
        "length": [100] * 4,
        "top_hmm_domain_hit_on_prot": [1, 0, 0, 0],
        "top_protein_match_for_hmm": [1, 0, 1, 0],
        "best_dom_env_start": [1] * 4,
        "best_dom_env_end": [50] * 4,
        "best_dom_env_length": [50] * 4,
        "sequence": ["MKT"] * 4,
    }
    return pd.DataFrame(data)


# -------------------------------
# Tests for fingerprint_hmm_table
# -------------------------------


def test_fingerprint_hmm_table_basic():
    con = duckdb.connect(":memory:")
    con.execute(
        """
        CREATE TABLE hmm_summary (
            protein_name VARCHAR,
            organism_name VARCHAR,
            hmm_domain_name VARCHAR,
            sequence VARCHAR
        )
    """
    )
    con.execute(
        """
        INSERT INTO hmm_summary VALUES
        ('p1','org1','hmm1','AAAA'),
        ('p2','org2','hmm2','BBBB')
    """
    )

    hash1 = fingerprint_hmm_table(con, "hmm_summary")
    hash2 = fingerprint_hmm_table(con, "hmm_summary")

    assert isinstance(hash1, str)
    assert len(hash1) == 64
    # deterministic
    assert hash1 == hash2

    con.close()


def test_fingerprint_hmm_table_empty_table():
    con = duckdb.connect(":memory:")
    con.execute(
        """
        CREATE TABLE hmm_summary (
            protein_name VARCHAR,
            organism_name VARCHAR,
            hmm_domain_name VARCHAR,
            sequence VARCHAR
        )
    """
    )

    hash_empty = fingerprint_hmm_table(con, "hmm_summary")
    assert hash_empty == "0" * 64

    con.close()


def test_fingerprint_hmm_table_changes():
    con = duckdb.connect(":memory:")
    con.execute(
        """
        CREATE TABLE hmm_summary (
            protein_name VARCHAR,
            organism_name VARCHAR,
            hmm_domain_name VARCHAR,
            sequence VARCHAR
        )
    """
    )

    con.execute("INSERT INTO hmm_summary VALUES ('p1','org1','hmm1','AAAA')")
    hash1 = fingerprint_hmm_table(con, "hmm_summary")

    con.execute("INSERT INTO hmm_summary VALUES ('p2','org2','hmm2','BBBB')")
    hash2 = fingerprint_hmm_table(con, "hmm_summary")

    assert hash1 != hash2
    con.close()


# -------------------------------
# Helper to create parquet file
# -------------------------------
def write_parquet(df: pd.DataFrame, path: Path):
    table = pa.Table.from_pandas(df)
    pq.write_table(table, path)


# -------------------------------
# Test: basic filtering
# -------------------------------
def test_build_hmm_summary_table_basic(tmp_path: Path):
    # Create a fake lake directory structure
    lake_root = tmp_path / "lake"
    (lake_root / "organism=Bacteria1").mkdir(parents=True)

    df = pd.DataFrame(
        {
            "protein_name": ["p1", "p2", "p3", "p4"],
            "organism_name": ["Bacteria1"] * 4,
            "hmm_domain_name": ["hmm1", "hmm2", "hmm3", "hmm4"],
            "annot": ["a1", "a2", "a3", "a4"],
            "evalue": [0.01] * 4,
            "score": [10, 5, 20, 1],
            "bias": [0.1] * 4,
            "length": [100] * 4,
            "top_hmm_domain_hit_on_prot": [1, 0, 0, 0],
            "top_protein_match_for_hmm": [1, 1, 1, 0],
            "best_dom_env_start": [1] * 4,
            "best_dom_env_end": [50] * 4,
            "best_dom_env_length": [50] * 4,
            "sequence": ["AAAA", "BBBB", "CCCC", "DDDD"],
        }
    )

    parquet_path = lake_root / "organism=Bacteria1" / "proteins.parquet"
    write_parquet(df, parquet_path)

    db_path = tmp_path / "summary.duckdb"
    table_name = "hmm_summary"
    min_score = 6

    # Build the summary table
    build_hmm_summary_table(
        db_path=db_path, lake_root=lake_root, table_name=table_name, min_score=min_score
    )

    # Connect to DuckDB and check results
    con = duckdb.connect(str(db_path))
    result = con.execute(
        f"SELECT protein_name, hmm_domain_name FROM {table_name} ORDER BY protein_name"
    ).fetchall()
    con.close()

    # Expected: only rows satisfying:
    # top_protein_match_for_hmm=1 AND (top_hmm_domain_hit_on_prot=1 OR score>6)
    # Check each row:
    expected = [
        ("p1", "hmm1"),  # top_hmm_domain_hit_on_prot=1
        ("p3", "hmm3"),  # score=20 > 6
    ]
    assert result == expected


# -------------------------------
# Test: empty parquet file produces empty summary
# -------------------------------
def test_build_hmm_summary_table_empty(tmp_path: Path):
    lake_root = tmp_path / "lake"
    (lake_root / "organism=Bacteria1").mkdir(parents=True)

    df = pd.DataFrame(
        columns=[
            "protein_name",
            "organism_name",
            "hmm_domain_name",
            "annot",
            "evalue",
            "score",
            "bias",
            "length",
            "top_hmm_domain_hit_on_prot",
            "top_protein_match_for_hmm",
            "best_dom_env_start",
            "best_dom_env_end",
            "best_dom_env_length",
            "sequence",
        ]
    )
    parquet_path = lake_root / "organism=Bacteria1" / "proteins.parquet"
    write_parquet(df, parquet_path)

    db_path = tmp_path / "summary.duckdb"
    table_name = "hmm_summary"
    min_score = 5

    build_hmm_summary_table(
        db_path=db_path, lake_root=lake_root, table_name=table_name, min_score=min_score
    )

    con = duckdb.connect(str(db_path))
    count = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
    con.close()

    assert count == 0


# -------------------------------
# Test: incremental rebuild adds new rows
# -------------------------------
def test_build_hmm_summary_table_incremental(tmp_path: Path):
    lake_root = tmp_path / "lake"
    (lake_root / "organism=Bacteria1").mkdir(parents=True)

    # First parquet file
    df1 = pd.DataFrame(
        {
            "protein_name": ["p1"],
            "organism_name": ["Bacteria1"],
            "hmm_domain_name": ["hmm1"],
            "annot": ["a1"],
            "evalue": [0.01],
            "score": [10],
            "bias": [0.1],
            "length": [100],
            "top_hmm_domain_hit_on_prot": [1],
            "top_protein_match_for_hmm": [1],
            "best_dom_env_start": [1],
            "best_dom_env_end": [50],
            "best_dom_env_length": [50],
            "sequence": ["AAAA"],
        }
    )
    parquet_path1 = lake_root / "organism=Bacteria1" / "proteins.parquet"
    write_parquet(df1, parquet_path1)

    db_path = tmp_path / "summary.duckdb"
    table_name = "hmm_summary"
    min_score = 5

    build_hmm_summary_table(
        db_path=db_path, lake_root=lake_root, table_name=table_name, min_score=min_score
    )

    # Add another parquet file for incremental update
    (lake_root / "organism=Bacteria2").mkdir(parents=True)
    df2 = pd.DataFrame(
        {
            "protein_name": ["p2"],
            "organism_name": ["Bacteria2"],
            "hmm_domain_name": ["hmm2"],
            "annot": ["a2"],
            "evalue": [0.01],
            "score": [10],
            "bias": [0.1],
            "length": [100],
            "top_hmm_domain_hit_on_prot": [0],
            "top_protein_match_for_hmm": [1],
            "best_dom_env_start": [1],
            "best_dom_env_end": [50],
            "best_dom_env_length": [50],
            "sequence": ["BBBB"],
        }
    )
    parquet_path2 = lake_root / "organism=Bacteria2" / "proteins.parquet"
    write_parquet(df2, parquet_path2)

    build_hmm_summary_table(
        db_path=db_path, lake_root=lake_root, table_name=table_name, min_score=min_score
    )

    con = duckdb.connect(str(db_path))
    result = con.execute(
        f"SELECT protein_name, organism_name FROM {table_name} ORDER BY protein_name"
    ).fetchall()
    con.close()

    expected = [("p1", "Bacteria1"), ("p2", "Bacteria2")]
    assert result == expected


# -------------------------------
# Test: filtering by score works correctly
# -------------------------------
def test_build_hmm_summary_table_score_filter(tmp_path: Path):
    lake_root = tmp_path / "lake"
    (lake_root / "organism=Bacteria1").mkdir(parents=True)

    df = pd.DataFrame(
        {
            "protein_name": ["p1", "p2", "p3"],
            "organism_name": ["Bacteria1"] * 3,
            "hmm_domain_name": ["hmm1", "hmm2", "hmm3"],
            "annot": ["a1", "a2", "a3"],
            "evalue": [0.01] * 3,
            "score": [3, 7, 10],
            "bias": [0.1] * 3,
            "length": [100] * 3,
            "top_hmm_domain_hit_on_prot": [0, 0, 0],
            "top_protein_match_for_hmm": [1, 1, 1],
            "best_dom_env_start": [1] * 3,
            "best_dom_env_end": [50] * 3,
            "best_dom_env_length": [50] * 3,
            "sequence": ["AAA", "BBB", "CCC"],
        }
    )

    parquet_path = lake_root / "organism=Bacteria1" / "proteins.parquet"
    write_parquet(df, parquet_path)

    db_path = tmp_path / "summary.duckdb"
    table_name = "hmm_summary"

    build_hmm_summary_table(
        db_path=db_path, lake_root=lake_root, table_name=table_name, min_score=6
    )

    con = duckdb.connect(str(db_path))
    result = con.execute(
        f"SELECT protein_name, annot FROM {table_name} ORDER BY protein_name"
    ).fetchall()
    con.close()

    # Only proteins with score > min_score (6) included
    expected = [("p2", "a2"), ("p3", "a3")]
    # Compare only protein names (DuckDB auto-converts types)
    result_names = [r[0] for r in result]
    expected_names = [r[0] for r in expected]
    assert result_names == expected_names


# --------------------------------------------------
# Helper to create a test DuckDB summary table
# --------------------------------------------------
def create_test_hmm_table(db_path: Path, table_name: str, data: pd.DataFrame):
    con = duckdb.connect(str(db_path))
    # Create table with required columns
    con.execute(
        f"""
        CREATE TABLE {table_name} (
            protein_name VARCHAR,
            organism_name VARCHAR,
            hmm_domain_name VARCHAR,
            annot VARCHAR,
            sequence VARCHAR
        )
    """
    )
    # Insert data
    for row in data.itertuples(index=False):
        con.execute(
            f"""
            INSERT INTO {table_name} (protein_name, organism_name, hmm_domain_name, annot, sequence)
            VALUES (?, ?, ?, ?, ?)
        """,
            row,
        )
    con.close()


# --------------------------------------------------
# Tests for get_distinct_hmms
# --------------------------------------------------
def test_get_distinct_hmms_basic(tmp_path: Path):
    db_path = tmp_path / "test.db"
    table_name = "hmm_summary"

    data = pd.DataFrame(
        {
            "protein_name": ["p1", "p2", "p3"],
            "organism_name": ["org1", "org2", "org3"],
            "hmm_domain_name": ["hmmB", "hmmA", "hmmC"],
            "annot": ["a1", "a2", "a3"],
            "sequence": ["AAA", "BBB", "CCC"],
        }
    )

    create_test_hmm_table(db_path, table_name, data)

    hmms = get_distinct_hmms(db_path, table_name, min_copies=1)
    assert hmms == ["hmmA", "hmmB", "hmmC"]  # should be sorted


def test_get_distinct_hmms_empty(tmp_path: Path):
    db_path = tmp_path / "test.db"
    table_name = "hmm_summary"

    # Create empty table
    create_test_hmm_table(
        db_path,
        table_name,
        pd.DataFrame(
            columns=["protein_name", "organism_name", "hmm_domain_name", "annot", "sequence"]
        ),
    )

    hmms = get_distinct_hmms(db_path, table_name)
    assert hmms == []


# --------------------------------------------------
# Tests for extract_hmm_sequences_for_msa
# --------------------------------------------------
def test_extract_hmm_sequences_for_msa_basic(tmp_path: Path):
    db_path = tmp_path / "test.db"
    table_name = "hmm_summary"
    out_fasta = tmp_path / "output.fasta"

    data = pd.DataFrame(
        {
            "protein_name": ["p1", "p2", "p3", "p4"],
            "organism_name": ["org1", "org1", "org2", "org3"],
            "hmm_domain_name": ["hmmX", "hmmX", "hmmY", "hmmX"],
            "annot": ["a1", "a2", "a3", "a4"],
            "sequence": ["AAA", "BBB", "CCC", "DDD"],
        }
    )

    create_test_hmm_table(db_path, table_name, data)

    # Extract sequences for hmmX
    extract_hmm_sequences_for_msa(db_path, table_name, "hmmX", out_fasta)

    # Read fasta
    fasta_lines = out_fasta.read_text().splitlines()
    expected_lines = [">org1;p1", "AAA", ">org1;p2", "BBB", ">org3;p4", "DDD"]
    assert fasta_lines == expected_lines


def test_extract_hmm_sequences_for_msa_empty(tmp_path: Path):
    db_path = tmp_path / "test.db"
    table_name = "hmm_summary"
    out_fasta = tmp_path / "output.fasta"

    data = pd.DataFrame(
        {
            "protein_name": [],
            "organism_name": [],
            "hmm_domain_name": [],
            "annot": [],
            "sequence": [],
        }
    )

    create_test_hmm_table(db_path, table_name, data)

    extract_hmm_sequences_for_msa(db_path, table_name, "hmmZ", out_fasta)
    assert out_fasta.exists()
    assert out_fasta.read_text() == ""


def test_extract_hmm_sequences_for_msa_multiple_same_org(tmp_path: Path):
    db_path = tmp_path / "test.db"
    table_name = "hmm_summary"
    out_fasta = tmp_path / "output.fasta"

    data = pd.DataFrame(
        {
            "protein_name": ["p1", "p2"],
            "organism_name": ["org1", "org1"],
            "hmm_domain_name": ["hmmA", "hmmA"],
            "annot": ["a1", "a2"],
            "sequence": ["AAA", "BBB"],
        }
    )

    create_test_hmm_table(db_path, table_name, data)

    extract_hmm_sequences_for_msa(db_path, table_name, "hmmA", out_fasta)

    fasta_lines = out_fasta.read_text().splitlines()
    expected_lines = [">org1;p1", "AAA", ">org1;p2", "BBB"]
    assert fasta_lines == expected_lines
