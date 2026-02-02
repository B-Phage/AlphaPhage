from pathlib import Path

import pandas as pd
import pytest
from data_contracts.hmm_lake import HMM_LAKE_PRIMARY_KEYS, HMM_LAKE_SCHEMA
from hmm.duckdb_io import write_hmm_results_with_sequences


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

    caplog.set_level("WARNING")
    write_hmm_results_with_sequences(empty_df, protein_block, organism_name, out_root)

    out_file = out_root / f"organism={organism_name}" / "proteins.parquet"
    # File should NOT be created
    assert not out_file.exists()

    # Warning should be logged
    assert any("No HMM hits to insert" in record.message for record in caplog.records)


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
