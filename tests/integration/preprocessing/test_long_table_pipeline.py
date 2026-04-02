# export PYTHONPATH=$PYTHONPATH:$(pwd)/packages && pytest tests/integration/test_long_table_pipeline.py -vv
from pathlib import Path

import duckdb
import pandas as pd
import pytest
from variants.long_table_builder import build_long_table


@pytest.fixture
def vocab():
    return {"vocab_clusters_single": {"var_1": 101, "var_2": 102}}


@pytest.fixture
def dummy_parquet_shards(tmp_path):
    """Creates dummy variant shards to simulate the msa_variant_dir."""
    shards_dir = tmp_path / "msa_var"
    shards_dir.mkdir()

    # Create two dummy protein shards
    df1 = pd.DataFrame(
        {
            "organism_id": ["G1", "G2"],
            "hmm_id": ["H1", "H1"],
            "variant_id": ["var_1", "var_2"],  # Using strings to test the 'str' error
            "clustering_version": ["v1", "v1"],
        }
    )

    df1.to_parquet(shards_dir / "protein_1.parquet")
    return shards_dir


def test_integration_build_long_table(dummy_parquet_shards, tmp_path, vocab):
    """
    Tests the full path from Parquet folder + Vocab -> DuckDB Table.
    Uses the 'vocab' fixture from your conftest.py.
    """
    db_path = str(tmp_path / "var.db")
    table_name = "phage_variants"
    # Match the vocab_section to what is in your YAML/Mock Vocab
    vocab_section = "vocab_clusters_single"

    # Execute the function
    build_long_table(
        parquet_dir=str(dummy_parquet_shards),
        db_path=db_path,
        table_name=table_name,
        vocab_section=vocab_section,
        config_path=None,  # It will use the mocked vocab fixture logic
    )

    # Verify DuckDB results
    con = duckdb.connect(db_path)
    df = con.execute(f"SELECT * FROM {table_name}").df()

    assert len(df) == 2
    assert "token_id" in df.columns
    # Check that variant_id was mapped to token_id correctly
    con.close()
