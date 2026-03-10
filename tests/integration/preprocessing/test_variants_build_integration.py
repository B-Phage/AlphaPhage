import os
import pathlib

import pandas as pd
import pytest
from omegaconf import OmegaConf
from variants.build import build_variants

# Path to real test data
TEST_MSA_FILE = "tests/data/phrog_1.csa"


@pytest.fixture
def vocab():
    """Minimal vocab configuration for integration tests"""
    return OmegaConf.create(
        {"filler_char": "-", "vocab_clusters_single": {str(i): i for i in range(20)}}
    )


@pytest.mark.integration
def test_build_variants_basic(tmp_path, vocab):
    """
    Verifies that the refactored build_variants produces a valid Parquet shard
    with the correct columns and metadata.
    """
    out_dir = tmp_path / "clusters"

    # Run the refactored build_variants
    build_variants(
        in_file=TEST_MSA_FILE,
        msa_cluster_dir=str(out_dir),
        vocab=vocab,
        reduction_method="diversity",
        num_clusters=4,
        matrix="identity",
    )

    # 1. Check file existence (Now .parquet instead of .mva/.mcd)
    expected_parquet = out_dir / "phrog_1.parquet"
    assert expected_parquet.exists()

    # Ensure the old sidecar files are NOT created (cleanup check)
    assert not (out_dir / "phrog_1.mva").exists()
    assert not (out_dir / "phrog_1.mcd").exists()

    # 2. Inspect Parquet data
    df = pd.read_parquet(expected_parquet)

    # 3. Validation: Check Schema and Content
    # These columns come from your reset_index() and hmm_id injection
    assert "organism_id" in df.columns
    assert "hmm_id" in df.columns
    assert "variant_id" in df.columns  # Enforced by your variant_lake_writer

    assert not df.empty
    assert (df["hmm_id"] == "phrog_1").all()

    # Check that cluster assignments are within the expected range
    assert df["variant_id"].nunique() <= 4
    assert df["variant_id"].dtype == "Int64"  # Per your enforce_schema logic


@pytest.mark.integration
def test_build_variants_with_X_characters(tmp_path, vocab):
    """
    Ensures that MSA files with 'X' residues are still processed into valid
    variant shards without crashing.
    """
    msa_path = tmp_path / "phrog_1_X.csa"

    # Inject 'X' into the test data
    with open(TEST_MSA_FILE, "r") as f_in, open(msa_path, "w") as f_out:
        for line in f_in:
            if "," in line:
                seq, rest = line.strip().split(",", 1)
                seq_modified = seq.replace("A", "X").replace("G", "X")
                f_out.write(f"{seq_modified},{rest}\n")
            else:
                f_out.write(line)

    out_dir = tmp_path / "clusters_X"

    build_variants(
        in_file=str(msa_path),
        msa_cluster_dir=str(out_dir),
        vocab=vocab,
        reduction_method="diversity",
        num_clusters=4,
        matrix="identity",
    )

    expected_parquet = out_dir / "phrog_1_X.parquet"
    assert expected_parquet.exists()

    df = pd.read_parquet(expected_parquet)
    # Ensure every organism still got an assignment despite the 'X' characters
    assert df["variant_id"].notna().all()


@pytest.mark.integration
def test_build_variants_no_reduction_integration(tmp_path, vocab):
    """
    Verifies that the pipeline runs correctly when reduction_method is None.
    """
    out_dir = tmp_path / "no_red"

    build_variants(
        in_file=TEST_MSA_FILE,
        msa_cluster_dir=str(out_dir),
        vocab=vocab,
        reduction_method=None,
        num_clusters=2,
        matrix="identity",
    )

    expected_parquet = out_dir / "phrog_1.parquet"
    assert expected_parquet.exists()

    df = pd.read_parquet(expected_parquet)
    assert "organism_id" in df.columns
    assert len(df["variant_id"].unique()) <= 2
