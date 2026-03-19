"""
Isolated run:
`pytest tests/unit/preprocessing/test_variant_clustering.py`
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from variants.clustering import (
    BLOSUM_FLOOR,
    IDENTITY_FLOOR,
    _calculate_radii,
    build_cluster_feature_table,
    perform_msa_clustering,
)

# -----------------------------------------------------------------------------
# 1. Unit Tests for _calculate_radii
# -----------------------------------------------------------------------------


def test_calculate_radii_singleton():
    """Ensures single-sequence clusters receive the fallback radius exactly."""
    ids = ["token_A"]
    dist_mtx = np.array([[0.0]])
    floor = 0.1
    result = _calculate_radii(ids, dist_mtx, floor)

    assert result["token_A"] == floor


def test_calculate_radii_identity_cluster():
    """Ensures multi-sequence identity clusters (dist 0.0) receive the fallback."""
    ids = ["token_A", "token_A"]
    dist_mtx = np.array([[0.0, 0.0], [0.0, 0.0]])
    floor = 0.1
    result = _calculate_radii(ids, dist_mtx, floor)

    assert result["token_A"] == floor


def test_calculate_radii_divergent_cluster():
    """Ensures clusters exceeding the floor return the actual max distance."""
    ids = ["token_B", "token_B"]
    # Max distance is 0.25, which is > floor of 0.1
    dist_mtx = np.array([[0.0, 0.25], [0.25, 0.0]])
    floor = 0.1
    result = _calculate_radii(ids, dist_mtx, floor)

    assert result["token_B"] == 0.25


def test_calculate_radii_mixed_clusters():
    """Verifies multiple unique IDs are handled correctly in one matrix."""
    ids = ["T1", "T1", "T2"]
    dist_mtx = np.array(
        [[0.0, 0.3, 0.8], [0.3, 0.0, 0.8], [0.8, 0.8, 0.0]]  # T1-T1 dist is 0.3  # T2 is singleton
    )
    floor = 0.1
    result = _calculate_radii(ids, dist_mtx, floor)

    assert result["T1"] == 0.3
    assert result["T2"] == floor


# -----------------------------------------------------------------------------
# 2. Unit Tests for perform_msa_clustering
# -----------------------------------------------------------------------------


@patch("variants.clustering.calculate_distance_matrix")
def test_perform_msa_clustering_single_row(mock_dist):
    """Tests the 'single_row_default' path."""
    df = pd.DataFrame([["A", "B", "C"]], columns=[0, 1, 2])
    mock_dist.return_value = np.array([[0.0]])

    labels, algo, k, mtx = perform_msa_clustering(df, 16, "identity")

    assert labels == ["1"]
    assert algo == "single_row_default"
    assert k == 1
    assert mtx.shape == (1, 1)


@patch("variants.clustering.calculate_distance_matrix")
def test_perform_msa_clustering_standard(mock_dist):
    """Tests standard agglomerative path with multi-sequence data."""
    df = pd.DataFrame([["A"], ["B"], ["C"]])
    # Mock a 3x3 distance matrix
    mock_dist.return_value = np.array([[0.0, 0.1, 0.5], [0.1, 0.0, 0.5], [0.5, 0.5, 0.0]])

    labels, algo, k, mtx = perform_msa_clustering(df, 2, "blosum62")

    assert len(labels) == 3
    assert algo == "agglomerative"
    assert k == 2
    assert isinstance(mtx, np.ndarray)


# -----------------------------------------------------------------------------
# 3. Integration Tests for build_cluster_feature_table
# -----------------------------------------------------------------------------


@patch("representation.encoding_methods.encode_onecol_vocab_vec")
@patch("variants.clustering.calculate_distance_matrix")
def test_build_cluster_feature_table_logic_flow(mock_dist, mock_encode):
    """
    CRITICAL TEST: Ensures that radii keys are synced with vocabulary tokens,
    fixing the 'Zero Radius' bug.
    """
    # Setup data
    df = pd.DataFrame({"p1": ["A", "A", "G"]}, index=["org1", "org2", "org3"])
    genomes = ["org1", "org2", "org3"]
    cols = ["p1"]
    vocab = {"1": 101, "2": 102}  # Label '1' maps to token 101

    # Mock distance matrix: org1&2 are identical, org3 is far
    mock_dist.return_value = np.array([[0.0, 0.0, 0.7], [0.0, 0.0, 0.7], [0.7, 0.7, 0.0]])

    # Mock encoding: Agglomerative puts org1&2 in '1', org3 in '2'
    # encode_onecol_vocab_vec returns [[token1, token2...]]
    mock_encode.return_value = [[101, 101, 102]]

    df_clusters, df_info, radii = build_cluster_feature_table(
        df, genomes, cols, 2, "identity", vocab
    )

    # Assertions
    # 1. Index must be strings of the tokens
    assert list(df_clusters.index) == ["101", "101", "102"]

    # 2. Radii keys must be strings of the tokens ("101"), not raw labels ("1")
    assert "101" in radii
    assert "102" in radii
    assert "1" not in radii

    # 3. Value check
    assert radii["101"] == IDENTITY_FLOOR  # Because internal dist was 0.0
    assert radii["102"] == IDENTITY_FLOOR  # Singleton


def test_build_cluster_feature_table_blosum_floor():
    """Verifies that blosum62 uses the correct floor constant."""
    df = pd.DataFrame({"p1": ["A", "R"]}, index=["o1", "o2"])
    vocab = {"1": 10, "2": 20}

    # Use real clustering/distance logic here (no mock) to check constant selection
    with patch("representation.encoding_methods.encode_onecol_vocab_vec") as mock_encode:
        mock_encode.return_value = [[10, 20]]
        _, _, radii = build_cluster_feature_table(df, ["o1", "o2"], ["p1"], 2, "blosum62", vocab)

        # Every value should be at least BLOSUM_FLOOR
        for r in radii.values():
            assert r >= BLOSUM_FLOOR
