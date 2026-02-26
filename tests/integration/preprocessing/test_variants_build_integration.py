import os
import shutil

import pandas as pd
import pytest
from omegaconf import OmegaConf
from variants.build import build_variants

# Path to real test data
TEST_MSA_FILE = "tests/data/phrog_1.csa"
TEST_MSA_PHROG2 = "tests/data/phrog_2.csa"
EXPECTED_MVA_PHROG2 = "tests/data/phrog_2.mva"


@pytest.fixture
def vocab():
    """Minimal vocab configuration for integration tests"""
    # Ensure vocab covers common residues + the filler
    return OmegaConf.create(
        {"filler_char": "-", "vocab_clusters_single": {str(i): i for i in range(20)}}
    )


def get_normalized_membership(path: str) -> pd.Series:
    """Reads the first column of a CSV and normalizes grouping labels.

    This follows your suggestion:
    1. Read CSV with no header and skip the first line.
    2. Extract the first column (Column 0).
    3. Map whatever characters exist there into a sequence of
       appearance (0, 1, 2...) so that the test is label-agnostic.
    """
    # Read raw data, ignoring headers
    df = pd.read_csv(path, header=None, skiprows=1)

    # Column 0 contains the cluster assignments (e.g., 4, 2, 4, 5...)
    raw_labels = df[0].tolist()

    mapping = {}
    normalized = []
    current_id = 0

    for label in raw_labels:
        if label not in mapping:
            mapping[label] = current_id
            current_id += 1
        normalized.append(mapping[label])

    return pd.Series(normalized)


@pytest.mark.integration
def test_build_variants_basic(tmp_path, vocab):
    """
    Full pipeline run with real MSA file using updated singular parameters.
    Checks outputs are created and contain valid data.
    """
    out_dir = tmp_path / "clusters"
    os.makedirs(out_dir, exist_ok=True)

    # Updated: reduction_methods -> reduction_method, removed lists for linkage/matrix
    build_variants(
        in_file=TEST_MSA_FILE,
        msa_cluster_dir=str(out_dir),
        vocab=vocab,
        reduction_method="diversity",
        num_clusters=4,
        matrix="identity",
    )

    # Check outputs
    expected_mva = out_dir / "phrog_1.mva"
    expected_mcd = out_dir / "phrog_1.mcd"
    assert expected_mva.exists()
    assert expected_mcd.exists()

    # Inspect data
    df_mva = pd.read_csv(expected_mva, index_col=0)
    df_mcd = pd.read_csv(expected_mcd, index_col=0)

    # Validation: Ensure dimensions are correct and not empty
    assert not df_mva.empty
    assert not df_mcd.empty

    # Check that we actually have clusters defined in the output
    # (Assuming the .mva file contains cluster assignments)
    assert len(df_mva.iloc[:, 0].unique()) <= 4


@pytest.mark.integration
def test_build_variants_with_X_characters(tmp_path, vocab):
    """
    Test pipeline behavior when MSA contains X characters (unknown residues).
    Ensures the refactored code handles character standardization with unknown residues.
    """
    # Copy the original MSA and inject X characters
    msa_path = tmp_path / "phrog_1_X.csa"

    # Use real file to create a variant with 'X'
    with open(TEST_MSA_FILE, "r") as f_in, open(msa_path, "w") as f_out:
        for line in f_in:
            if "," in line:
                seq, rest = line.strip().split(",", 1)
                # Inject 'X' into the sequence
                seq_modified = seq.replace("A", "X").replace("G", "X")
                f_out.write(f"{seq_modified},{rest}\n")
            else:
                f_out.write(line)

    out_dir = tmp_path / "clusters_X"
    os.makedirs(out_dir, exist_ok=True)

    # Updated to singular parameters
    build_variants(
        in_file=str(msa_path),
        msa_cluster_dir=str(out_dir),
        vocab=vocab,
        reduction_method="diversity",
        num_clusters=4,
        matrix="identity",
    )

    expected_mva = out_dir / "phrog_1_X.mva"
    assert expected_mva.exists()

    # Verify 'X' characters handled or preserved in the variant table
    df_mva = pd.read_csv(expected_mva, index_col=0)
    # Check if 'X' is present in the sequence data column (adjusting for your specific .mva schema)
    # This assumes the sequence or part of it is stored as a string in the columns
    mva_content = df_mva.values.astype(str).flatten()
    assert any("X" in cell for cell in mva_content)


@pytest.mark.integration
def test_build_variants_no_reduction_integration(tmp_path, vocab):
    """
    Integration check: ensure the pipeline runs without any reduction method specified.
    """
    out_dir = tmp_path / "no_red"
    os.makedirs(out_dir, exist_ok=True)

    build_variants(
        in_file=TEST_MSA_FILE,
        msa_cluster_dir=str(out_dir),
        vocab=vocab,
        reduction_method=None,  # Specifically testing the None path
        num_clusters=2,
        matrix="identity",
    )

    expected_mva = out_dir / "phrog_1.mva"
    assert expected_mva.exists()
    df_mva = pd.read_csv(expected_mva, index_col=0)
    assert not df_mva.empty


@pytest.mark.integration
def test_build_variants_regression_phrog2_membership(tmp_path, vocab):
    """
    Regression test for phrog_2.csa groupings.
    """
    out_dir = tmp_path / "regression_membership"
    os.makedirs(out_dir, exist_ok=True)

    # 1. Run the build
    build_variants(
        in_file="tests/data/phrog_2.csa", msa_cluster_dir=str(out_dir), vocab=vocab, num_clusters=16
    )

    generated_mva_path = out_dir / "phrog_2.mva"
    expected_mva_path = "tests/data/phrog_2.mva"

    # 2. Extract normalized memberships
    norm_generated = get_normalized_membership(str(generated_mva_path))
    norm_expected = get_normalized_membership(expected_mva_path)

    # 3. Compare grouping structure
    # NOTE: This will still fail if row counts (70 vs 69) don't match!
    pd.testing.assert_series_equal(norm_generated, norm_expected, obj="Cluster Grouping Structure")
