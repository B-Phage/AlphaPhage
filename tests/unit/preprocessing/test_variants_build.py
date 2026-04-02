import argparse
import pathlib
import sys
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from omegaconf import OmegaConf
from variants.build import build_variants, cli


@pytest.fixture
def vocab():
    """Minimal vocab fixture for tests"""
    # Note: filler_char is still used for standardization, even if vocab_dict is gone from clustering
    return OmegaConf.create({"filler_char": "-", "vocab_clusters_single": {"A": 0, "B": 1}})


@pytest.fixture
def mock_df():
    """Minimal DataFrame returned by process_msa_table"""
    return pd.DataFrame(
        [["A", "C"], ["G", "T"]], index=["genomeA", "genomeB"], columns=["pos1", "pos2"]
    )


@patch("variants.build.write_variant_shard")
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
    Verifies that build_variants correctly constructs the 6-column shard_df
    using raw string labels and passes it to the lake writer.
    """
    # 1. Setup Input State
    mock_process.return_value = mock_df
    mock_standardize.return_value = (mock_df, None)
    mock_diversity.return_value = [["pos1"]]

    # FIX: Use string indices to match perform_msa_clustering output
    mock_clusters_df = pd.DataFrame(index=["0", "1"])
    mock_radii = {"0": 0.05, "1": 0.05}
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

    # 3. Assertions
    args, _ = mock_write.call_args
    passed_df = args[0]

    # Verify All 6 Columns now present in the "Gallery" shard
    expected_cols = [
        "organism_id",
        "variant_id",
        "hmm_id",
        "sequence_aligned",
        "radius",
        "clustering_version",
    ]
    assert all(col in passed_df.columns for col in expected_cols)

    # Verify Data Types and Values
    assert passed_df["variant_id"].tolist() == ["0", "1"]
    assert isinstance(passed_df["variant_id"].iloc[0], str)
    assert passed_df["radius"].tolist() == [0.05, 0.05]

    # Verify sequence reconstruction (A+C = AC, G+T = GT)
    assert passed_df["sequence_aligned"].tolist() == ["AC", "GT"]


def test_build_variants_no_reduction_version_string(tmp_path, vocab, mock_df):
    out_dir = tmp_path / "clusters"
    in_file = tmp_path / "test.csa"
    in_file.write_text("dummy")

    with (
        patch("variants.build.process_msa_table") as mock_process,
        patch("variants.build.standardize_frame_chars") as mock_std,
        patch("variants.build.build_cluster_feature_table") as mock_build,
        patch("variants.build.write_variant_shard") as mock_write,
    ):
        mock_process.return_value = mock_df
        mock_std.return_value = (mock_df, None)
        # FIX: Use string labels "0" and "1"
        mock_build.return_value = (
            pd.DataFrame(index=["0", "1"]),
            pd.DataFrame(),
            {"0": 0.1, "1": 0.1},
        )

        build_variants(
            in_file=str(in_file),
            msa_cluster_dir=str(out_dir),
            vocab=vocab,
            reduction_method=None,
            num_clusters=8,
            matrix="identity",
        )

        passed_df = mock_write.call_args[0][0]
        assert "red-none" in passed_df["clustering_version"].iloc[0]
        assert passed_df["variant_id"].tolist() == ["0", "1"]


## --- CLI and Entry Point Tests ---


@patch("variants.build.get_vocab")
@patch("variants.build.build_variants")
@patch("argparse.ArgumentParser.parse_args")
def test_cli_execution_logic(mock_parse, mock_build, mock_get_vocab, vocab):
    """Tests that CLI correctly parses arguments and passes them to build_variants."""
    # 1. Setup Mock Args
    mock_args = MagicMock()
    mock_args.in_file = "input.csa"
    mock_args.msa_cluster_dir = "output_dir"
    mock_args.num_clusters = 16
    mock_args.reduction_method = "diversity"
    mock_args.matrix = "blosum62"
    mock_parse.return_value = mock_args

    mock_get_vocab.return_value = vocab

    # 2. Call CLI
    cli()

    # 3. Assertions
    mock_build.assert_called_once_with(
        in_file="input.csa",
        msa_cluster_dir="output_dir",
        vocab=vocab,
        reduction_method="diversity",
        num_clusters=16,
        matrix="blosum62",
    )


def test_cli_parser_integration():
    """Tests the actual argparse integration using sys.argv patching."""
    test_args = [
        "prog",
        "-f",
        "test.csa",
        "-o",
        "out_dir",
        "-c",
        "32",
        "-r",
        "diversity",
        "-m",
        "blosum62",
    ]

    with (
        patch.object(sys, "argv", test_args),
        patch("variants.build.get_vocab") as mock_get_vocab,
        patch("variants.build.build_variants") as mock_build,
    ):

        mock_get_vocab.return_value = {"dummy": "vocab"}

        cli()

        # Verify the types were correctly converted (e.g., num_clusters to int)
        args, kwargs = mock_build.call_args
        assert kwargs["num_clusters"] == 32
        assert kwargs["matrix"] == "blosum62"
        assert kwargs["in_file"] == "test.csa"


def test_build_variants_radius_lookup_failure(tmp_path, vocab, mock_df):
    """Covers the 'raise' block in the radius lookup try/except."""
    out_dir = tmp_path / "clusters"
    in_file = tmp_path / "test.csa"
    in_file.write_text("dummy")

    with (
        patch("variants.build.process_msa_table") as mock_p,
        patch("variants.build.standardize_frame_chars") as mock_s,
        patch("variants.build.build_cluster_feature_table") as mock_b,
    ):

        mock_p.return_value = mock_df
        mock_s.return_value = (mock_df, None)

        # Return a cluster label "2" which is NOT in our radii dict
        mock_b.return_value = (
            pd.DataFrame(index=["2"]),
            pd.DataFrame(),
            {"0": 0.1},  # Missing key "2"
        )

        with pytest.raises(KeyError):
            build_variants(in_file=str(in_file), msa_cluster_dir=str(out_dir), vocab=vocab)
