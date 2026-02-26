import os
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
    """Minimal DataFrame returned by process_msa_table"""
    return pd.DataFrame([[1, 2], [3, 4]], index=["genomeA", "genomeB"], columns=["pos1", "pos2"])


@patch("variants.build.build_cluster_feature_table")
@patch("variants.build.split_names_diversity")
@patch("variants.build.standardize_frame_chars")
@patch("variants.build.process_msa_table")
def test_build_variants_basic(
    mock_process, mock_standardize, mock_diversity, mock_build_cluster, tmp_path, vocab, mock_df
):
    """
    Test main orchestration logic of build_variants with singular parameters.
    """
    # Setup mocks
    mock_process.return_value = mock_df
    mock_standardize.return_value = (mock_df, None)
    mock_diversity.return_value = [["pos1"]]
    mock_build_cluster.return_value = (MagicMock(), mock_df)

    out_dir = tmp_path / "clusters"
    os.makedirs(out_dir, exist_ok=True)

    in_file = tmp_path / "test.csa"
    in_file.write_text("dummy content")

    build_variants(
        in_file=str(in_file),
        msa_cluster_dir=str(out_dir),
        vocab=vocab,
        reduction_method="diversity",
        num_clusters=8,
        matrix="blosum62",
    )

    # Verify mocks called
    mock_process.assert_called_once_with(str(in_file), vocab=vocab)
    mock_standardize.assert_called_once()
    mock_diversity.assert_called_once_with(mock_df)
    mock_build_cluster.assert_called_once()

    # Check that output files are created
    expected_mva = out_dir / "test.mva"
    expected_mcd = out_dir / "test.mcd"
    assert expected_mva.exists()
    assert expected_mcd.exists()


@patch("variants.build.build_cluster_feature_table")
@patch("variants.build.standardize_frame_chars")
@patch("variants.build.process_msa_table")
def test_build_variants_no_reduction(
    mock_process, mock_standardize, mock_build_cluster, tmp_path, vocab, mock_df
):
    """
    Test build_variants when no reduction_method is provided.
    """
    mock_process.return_value = mock_df
    mock_standardize.return_value = (mock_df, None)
    mock_build_cluster.return_value = (MagicMock(), mock_df)

    out_dir = tmp_path / "clusters"
    in_file = tmp_path / "test.csa"
    in_file.write_text("dummy")

    with patch("variants.build.split_names_diversity") as mock_div:
        build_variants(str(in_file), str(out_dir), vocab=vocab, reduction_method=None)
        mock_div.assert_not_called()


@patch("variants.build.build_cluster_feature_table")
@patch("variants.build.split_names_diversity")
@patch("variants.build.standardize_frame_chars")
@patch("variants.build.process_msa_table")
def test_build_variants_empty_columns_fallback(
    mock_process, mock_standardize, mock_div, mock_build, tmp_path, vocab
):
    """
    Test that build_variants handles case where diversity returns empty list.
    """
    df_single = pd.DataFrame([[1]], index=["g1"], columns=["p1"])
    mock_process.return_value = df_single
    mock_standardize.return_value = (df_single, None)
    mock_div.return_value = []  # Edge case: diversity returns nothing
    mock_build.return_value = (MagicMock(), df_single)

    out_dir = tmp_path / "clusters"
    in_file = tmp_path / "test.csa"
    in_file.write_text("dummy")

    build_variants(str(in_file), str(out_dir), vocab=vocab, reduction_method="diversity")

    # Should fall back to df.columns
    args, kwargs = mock_build.call_args
    assert kwargs["columns_subset"] == ["p1"]


@patch("variants.build.os.makedirs")
@patch("variants.build.build_cluster_feature_table")
@patch("variants.build.standardize_frame_chars")
@patch("variants.build.process_msa_table")
def test_build_variants_directory_creation(
    mock_process, mock_standardize, mock_build_cluster, mock_makedirs, tmp_path, vocab, mock_df
):
    """
    Test that the output directory is created via os.makedirs
    """
    mock_process.return_value = mock_df
    mock_standardize.return_value = (mock_df, None)
    mock_build_cluster.return_value = (MagicMock(), mock_df)

    out_dir = tmp_path / "new_dir"
    in_file = tmp_path / "test.csa"
    in_file.write_text("dummy")

    build_variants(str(in_file), str(out_dir), vocab=vocab)
    mock_makedirs.assert_called_once_with(str(out_dir), exist_ok=True)


def test_cli_invokes_build_variants(tmp_path):
    """
    Test the CLI function calls build_variants with correct singular arguments.
    """
    in_file = tmp_path / "input.csa"
    in_file.write_text("dummy content")
    out_dir = tmp_path / "out"

    # Patch get_vocab and build_variants
    with (
        patch("variants.build.get_vocab") as mock_get_vocab,
        patch("variants.build.build_variants") as mock_build,
        patch("argparse.ArgumentParser.parse_args") as mock_parse,
    ):

        mock_get_vocab.return_value = MagicMock()
        mock_parse.return_value = argparse.Namespace(
            in_file=str(in_file),
            msa_cluster_dir=str(out_dir),
            num_clusters=8,
            reduction_method="diversity",
            matrix="identity",
        )

        cli()

        mock_build.assert_called_once_with(
            in_file=str(in_file),
            msa_cluster_dir=str(out_dir),
            vocab=mock_get_vocab.return_value,
            reduction_method="diversity",
            num_clusters=8,
            matrix="identity",
        )


import argparse


def test_cli_error_on_missing_required(tmp_path):
    """
    Test that CLI fails when required arguments are missing.
    The new 'required=True' in argparse will handle this via SystemExit.
    """
    with patch("sys.argv", ["prog", "-o", "some_dir"]):
        with pytest.raises(SystemExit):
            cli()
