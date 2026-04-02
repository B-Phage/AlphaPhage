"""
Isolated run:
pytest tests/unit/preprocessing/test_variant_clustering.py
"""

from unittest.mock import patch

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
    ids = ["0"]
    dist_mtx = np.array([[0.0]])
    floor = 0.1
    result = _calculate_radii(ids, dist_mtx, floor)

    assert result["0"] == floor


def test_calculate_radii_identity_cluster():
    """Ensures multi-sequence identity clusters (dist 0.0) receive the fallback."""
    ids = ["0", "0"]
    dist_mtx = np.array([[0.0, 0.0], [0.0, 0.0]])
    floor = 0.1
    result = _calculate_radii(ids, dist_mtx, floor)

    assert result["0"] == floor


def test_calculate_radii_divergent_cluster():
    """Ensures clusters exceeding the floor return the actual max distance."""
    ids = ["1", "1"]
    # Max distance is 0.25, which is > floor of 0.1
    dist_mtx = np.array([[0.0, 0.25], [0.25, 0.0]])
    floor = 0.1
    result = _calculate_radii(ids, dist_mtx, floor)

    assert result["1"] == 0.25


def test_calculate_radii_mixed_clusters():
    """Verifies multiple unique IDs are handled correctly in one matrix."""
    ids = ["0", "0", "1"]
    dist_mtx = np.array([[0.0, 0.3, 0.8], [0.3, 0.0, 0.8], [0.8, 0.8, 0.0]])
    floor = 0.1
    result = _calculate_radii(ids, dist_mtx, floor)

    assert result["0"] == 0.3
    assert result["1"] == floor


# -----------------------------------------------------------------------------
# 2. Unit Tests for perform_msa_clustering
# -----------------------------------------------------------------------------


@patch("variants.clustering.calculate_distance_matrix")
def test_perform_msa_clustering_single_row(mock_dist):
    """Tests the 'single_row_default' path (num_unique <= 1)."""
    df = pd.DataFrame([["A", "B", "C"]], columns=[0, 1, 2])
    mock_dist.return_value = np.array([[0.0]])

    labels, algo, k, mtx = perform_msa_clustering(df, 16, "identity")

    assert labels == ["0"]
    assert algo == "single_row_default"
    assert k == 1
    assert mtx.shape == (1, 1)


@patch("variants.clustering.calculate_distance_matrix")
def test_perform_msa_clustering_standard(mock_dist):
    """Tests standard agglomerative path with multi-sequence data."""
    df = pd.DataFrame([["A"], ["B"], ["C"]])
    # Mock a 3x3 distance matrix where index 0 and 1 are close
    mock_dist.return_value = np.array([[0.0, 0.1, 0.8], [0.1, 0.0, 0.8], [0.8, 0.8, 0.0]])

    labels, algo, k, mtx = perform_msa_clustering(df, 2, "blosum62")

    assert len(labels) == 3
    assert algo == "agglomerative"
    assert k == 2
    # Check that labels grouped 0 and 1 together (should both be '1' or both '0')
    assert labels[0] == labels[1]
    assert labels[0] != labels[2]


# -----------------------------------------------------------------------------
# 3. Integration Tests for build_cluster_feature_table
# -----------------------------------------------------------------------------


@patch("variants.clustering.calculate_distance_matrix")
def test_build_cluster_feature_table_logic_flow(mock_dist):
    """
    Ensures that radii keys match the DataFrame index and use raw labels.
    """
    # Setup data
    df = pd.DataFrame({"p1": ["A", "A", "G"]}, index=["org1", "org2", "org3"])
    genomes = ["org1", "org2", "org3"]
    cols = ["p1"]

    # Mock distance matrix: org1&2 are identical, org3 is far
    mock_dist.return_value = np.array([[0.0, 0.0, 0.7], [0.0, 0.0, 0.7], [0.7, 0.7, 0.0]])

    df_clusters, df_info, radii = build_cluster_feature_table(df, genomes, cols, 2, "identity")

    # 1. Index must be strings of the raw cluster labels
    # Agglomerative with 2 clusters on this matrix will yield ["1", "1", "0"] or ["0", "0", "1"]
    unique_labels = list(df_clusters.index.unique())
    assert len(unique_labels) == 2
    assert all(isinstance(label, str) for label in df_clusters.index)

    # 2. Radii keys must match the index strings
    for label in unique_labels:
        assert label in radii

    # 3. Metadata check
    assert df_info.loc[0, "clustering_algorithm"] == "agglomerative"
    assert df_info.loc[0, "msa_width"] == 1


@patch("variants.clustering.calculate_distance_matrix")
def test_build_cluster_feature_table_blosum_floor(mock_dist):
    """Verifies that blosum62 uses the BLOSUM_FLOOR constant."""
    df = pd.DataFrame({"p1": ["A", "R"]}, index=["o1", "o2"])
    # Mock very small distance to trigger the floor
    mock_dist.return_value = np.array([[0.0, 0.001], [0.001, 0.0]])

    _, _, radii = build_cluster_feature_table(df, ["o1", "o2"], ["p1"], 2, "blosum62")

    # Every value should be the BLOSUM_FLOOR because 0.001 < 0.10
    for r in radii.values():
        assert r == BLOSUM_FLOOR
