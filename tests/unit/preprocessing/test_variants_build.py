import argparse
import os
import pathlib
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from omegaconf import OmegaConf
from variants.build import build_variants, cli


@pytest.fixture
def vocab():
    """Minimal vocab fixture for tests"""
    return OmegaConf.create({"filler_char": "-", "vocab_clusters_single": {"A": 0, "B": 1}})


@pytest.fixture
def mock_df():
    """Minimal DataFrame returned by process_msa_table - now with strings"""
    return pd.DataFrame(
        [["A", "C"], ["G", "T"]], index=["genomeA", "genomeB"], columns=["pos1", "pos2"]
    )


@patch("variants.build.write_variant_shard")  # New: Mock the lake writer
@patch("variants.build.build_cluster_feature_table")
@patch("variants.build.split_names_diversity")
@patch("variants.build.standardize_frame_chars")
@patch("variants.build.process_msa_table")
def test_build_variants_orchestration(
    mock_process,
    mock_standardize,
    mock_diversity,
    mock_build_cluster,
    mock_write,
    tmp_path,
    vocab,
    mock_df,
):
    """
    Verifies that build_variants correctly constructs the 4-column shard_df
    and passes it to the lake writer.
    """
    # 1. Setup Input State
    mock_process.return_value = mock_df
    mock_standardize.return_value = (mock_df, None)
    mock_diversity.return_value = [["pos1"]]

    # Mock clustering result: Index is the variant assignments (e.g. cluster 5 and 6)
    mock_clusters_df = pd.DataFrame({"AA_data": ["X", "Y"]}, index=[5, 6])
    mock_radii = {5: 0.1, 6: 0.1}  # Added mock radii
    mock_build_cluster.return_value = (mock_clusters_df, pd.DataFrame(), mock_radii)

    out_dir = tmp_path / "clusters"
    in_file = tmp_path / "phrog_1.csa"
    in_file.write_text("dummy")

    # 2. Execute
    build_variants(
        in_file=str(in_file),
        msa_cluster_dir=str(out_dir),
        vocab=vocab,
        reduction_method="diversity",
        num_clusters=8,
        matrix="identity",
    )

    # 3. Assertions on the Data Construction
    # Get the dataframe that was passed to write_variant_shard
    args, _ = mock_write.call_args
    passed_df = args[0]
    passed_path = args[1]

    # Verify Column Structure
    expected_cols = ["organism_id", "variant_id", "hmm_id", "clustering_version"]
    assert all(col in passed_df.columns for col in expected_cols)

    # Verify Data Alignment
    assert passed_df["organism_id"].tolist() == ["genomeA", "genomeB"]
    assert passed_df["variant_id"].tolist() == [5, 6]
    assert passed_df["hmm_id"].unique()[0] == "phrog_1"

    # Verify Clustering Version String
    expected_ver = "red-diversity_clusters-8_mat-identity"
    assert passed_df["clustering_version"].unique()[0] == expected_ver

    # Verify Path Logic
    assert passed_path == pathlib.Path(out_dir) / "phrog_1.parquet"
    from unittest.mock import MagicMock, patch


def test_build_variants_no_reduction_version_string(tmp_path, vocab, mock_df):
    """
    Using context managers to avoid fixture/mock argument collisions.
    """
    # 1. Setup paths
    out_dir = tmp_path / "clusters"
    in_file = tmp_path / "test.csa"
    in_file.write_text("dummy")

    # 2. Patch everything inside the 'with' block
    with (
        patch("variants.build.process_msa_table") as mock_process,
        patch("variants.build.standardize_frame_chars") as mock_std,
        patch("variants.build.build_cluster_feature_table") as mock_build,
        patch("variants.build.write_variant_shard") as mock_write,
    ):

        # Setup return values
        mock_process.return_value = mock_df
        mock_std.return_value = (mock_df, [0, 0])
        # Mock clustering: index is [1, 2] (the variants)
        mock_build.return_value = (pd.DataFrame(index=[1, 2]), pd.DataFrame(), {1: 0.1, 2: 0.1})

        # 3. Execute
        build_variants(
            in_file=str(in_file),
            msa_cluster_dir=str(out_dir),
            vocab=vocab,
            reduction_method=None,
            num_clusters=8,
            matrix="identity",
        )

        # 4. Assert
        # Get the dataframe passed to write_variant_shard (first arg of first call)
        passed_df = mock_write.call_args[0][0]

        # Verify the 'none' logic in the version string
        assert "red-none" in passed_df["clustering_version"].iloc[0]
        assert passed_df["variant_id"].tolist() == [1, 2]
        assert passed_df["organism_id"].tolist() == ["genomeA", "genomeB"]


def test_cli_invokes_build_variants(tmp_path):
    """
    Ensures CLI still maps arguments correctly to the build function.
    """
    in_file = tmp_path / "input.csa"
    in_file.write_text("dummy")
    out_dir = tmp_path / "out"

    with (
        patch("variants.build.get_vocab") as mock_get_vocab,
        patch("variants.build.build_variants") as mock_build,
        patch("argparse.ArgumentParser.parse_args") as mock_parse,
    ):
        mock_parse.return_value = argparse.Namespace(
            in_file=str(in_file),
            msa_cluster_dir=str(out_dir),
            num_clusters=16,
            reduction_method="diversity",
            matrix="identity",
        )
        cli()
        mock_build.assert_called_once()
