"""
Isolated run:
`pytest tests/unit/preprocessing/test_variant_clustering.py`
"""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from variants.clustering import (
    build_cluster_feature_table,
    cluster_aggregate,
    cluster_sequences_agglomerate,
)

# =========================================================
# Fixtures
# =========================================================


@pytest.fixture
def simple_alignment_df():
    """
    3 genomes, 3 positions.
    Two genomes identical, one different.
    """
    return pd.DataFrame(
        [
            ["A", "A", "A"],
            ["A", "A", "A"],
            ["B", "B", "B"],
        ],
        index=["g1", "g2", "g3"],
        columns=["pos1", "pos2", "pos3"],
    )


@pytest.fixture
def vocab_dict():
    return {"1": 1, "2": 2}


@pytest.fixture
def mock_distance_matrix():
    """
    Deterministic symmetric matrix:
    g1 and g2 close, g3 far.
    """
    return np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ]
    )


# =========================================================
# cluster_sequences_agglomerate
# =========================================================


def test_cluster_sequences_basic(mock_distance_matrix):
    clusters = cluster_sequences_agglomerate(
        mock_distance_matrix,
        n_clusters=2,
        linkage="average",
    )

    assert len(clusters) == 3
    assert set(clusters).issubset({"0", "1"})


def test_cluster_sequences_reduces_if_too_many(mock_distance_matrix):
    clusters = cluster_sequences_agglomerate(
        mock_distance_matrix,
        n_clusters=10,
        linkage="average",
    )

    assert len(clusters) == 3


def test_cluster_sequences_invalid_cluster_number(mock_distance_matrix):
    with pytest.raises(ValueError):
        cluster_sequences_agglomerate(
            mock_distance_matrix,
            n_clusters=1,
        )


# =========================================================
# cluster_aggregate
# =========================================================


@patch("variants.distance.calculate_distance_matrix")
def test_cluster_aggregate_happy_path(mock_calc, simple_alignment_df, mock_distance_matrix):
    mock_calc.return_value = mock_distance_matrix

    series = cluster_aggregate(
        alignment=simple_alignment_df.values,
        index=list(simple_alignment_df.index),
        matrix="identity",
        num_clusters=2,
    )

    assert isinstance(series, pd.Series)
    assert list(series.index) == ["g1", "g2", "g3"]
    assert set(series.values).issubset({"0", "1"})


def test_cluster_aggregate_invalid_matrix(simple_alignment_df):
    with pytest.raises(ValueError):
        cluster_aggregate(
            alignment=simple_alignment_df.values,
            index=list(simple_alignment_df.index),
            matrix="invalid",
            num_clusters=2,
        )


def test_cluster_aggregate_invalid_clusters(simple_alignment_df):
    with pytest.raises(ValueError):
        cluster_aggregate(
            alignment=simple_alignment_df.values,
            index=list(simple_alignment_df.index),
            matrix="identity",
            num_clusters=1,
        )


def test_cluster_aggregate_index_length_mismatch(simple_alignment_df):
    with pytest.raises(ValueError):
        cluster_aggregate(
            alignment=simple_alignment_df.values,
            index=["g1", "g2"],  # wrong length
            matrix="identity",
            num_clusters=2,
        )


# ---------------------------------------------------------
# Test non-ndarray input triggers conversion
# ---------------------------------------------------------
@patch("variants.distance.calculate_distance_matrix")
def test_cluster_aggregate_list_input_triggers_ndarray(mock_calc):
    """
    Ensure that passing a list-of-lists alignment
    triggers the internal conversion to np.ndarray.
    """
    # alignment as a plain list-of-lists (not ndarray)
    alignment_list = [
        ["A", "A", "A"],
        ["A", "A", "A"],
        ["B", "B", "B"],
    ]
    index = ["g1", "g2", "g3"]

    # mock distance matrix for deterministic behavior
    mock_calc.return_value = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ]
    )

    series = cluster_aggregate(
        alignment=alignment_list,
        index=index,
        matrix="identity",
        num_clusters=2,
    )

    # Ensure output is correct type
    assert isinstance(series, pd.Series)
    assert list(series.index) == index
    assert set(series.values).issubset({"0", "1"})


# =========================================================
# build_cluster_feature_table
# =========================================================


@patch("variants.distance.calculate_distance_matrix")
def test_build_cluster_feature_table_basic(
    mock_calc,
    simple_alignment_df,
    vocab_dict,  # If this is a fixture, you might need to override it locally
    mock_distance_matrix,
):
    mock_calc.return_value = mock_distance_matrix

    # OVERRIDE/ENSURE vocab_dict has 1-based string keys
    # Your current error shows vocab_dict = {'0': 0, '1': 1}
    # Change it to support '1' and '2'
    test_vocab = {"1": 1, "2": 2}

    df_clusters, df_info = build_cluster_feature_table(
        df=simple_alignment_df,
        genomes=list(simple_alignment_df.index),
        columns_subset=["pos1", "pos2", "pos3"],
        num_clusters=2,
        matrix="identity",
        vocab_dict=test_vocab,  # Use the updated vocab here
    )

    # --- Check clusters dataframe ---
    assert isinstance(df_clusters, pd.DataFrame)
    assert list(df_clusters.columns) == list(simple_alignment_df.columns)
    assert len(df_clusters) == 3

    # Index should be cluster token combinations
    assert all(isinstance(i, str) for i in df_clusters.index)

    # --- Check metadata ---
    assert df_info.iloc[0]["feature_type"] == "cluster"
    assert df_info.iloc[0]["clustering_algorithm"] == "agglomerative"
    assert df_info.iloc[0]["distance_metric"] == "identity"
    assert df_info.iloc[0]["num_clusters"] == 2
    assert df_info.iloc[0]["msa_width"] == 3


@patch("variants.distance.calculate_distance_matrix")
def test_build_cluster_feature_table_reduces_clusters(
    mock_calc,
    simple_alignment_df,
    vocab_dict,  # This fixture is likely providing {'0': 0, '1': 1}
    mock_distance_matrix,
):
    mock_calc.return_value = mock_distance_matrix

    # Fix the vocab to include '2'
    local_vocab = {"1": 1, "2": 2}

    df_clusters, df_info = build_cluster_feature_table(
        df=simple_alignment_df,
        genomes=list(simple_alignment_df.index),
        columns_subset=["pos1", "pos2", "pos3"],
        num_clusters=10,  # too many
        matrix="identity",
        vocab_dict=local_vocab,  # Use the fixed vocab
    )

    # --- Assertions ---
    # The dataset has exactly 2 unique sequences, so it must be exactly 2
    assert df_info.iloc[0]["num_clusters"] == 2

    # Verify the index contains our encoded tokens (1 or 2)
    assert set(df_clusters.index).issubset({"1", "2"})


def test_build_cluster_feature_table_invalid_matrix(simple_alignment_df, vocab_dict):
    with pytest.raises(ValueError):
        build_cluster_feature_table(
            df=simple_alignment_df,
            genomes=list(simple_alignment_df.index),
            columns_subset=["pos1"],
            num_clusters=2,
            matrix="invalid",
            vocab_dict=vocab_dict,
        )


def test_build_cluster_feature_table_invalid_cluster_type(simple_alignment_df, vocab_dict):
    with pytest.raises(TypeError):
        build_cluster_feature_table(
            df=simple_alignment_df,
            genomes=list(simple_alignment_df.index),
            columns_subset=["pos1"],
            num_clusters="two",  # type: ignore
            matrix="identity",
            vocab_dict=vocab_dict,
        )


def test_build_cluster_feature_table_cluster_too_small(simple_alignment_df, vocab_dict):
    with pytest.raises(ValueError):
        build_cluster_feature_table(
            df=simple_alignment_df,
            genomes=list(simple_alignment_df.index),
            columns_subset=["pos1"],
            num_clusters=1,
            matrix="identity",
            vocab_dict=vocab_dict,
        )


@pytest.fixture
def mock_vocab():
    """Provides a minimal vocab dictionary for token encoding."""
    # This simulates the vocab_clusters_single structure
    return {"1": 101, "2": 102, "gap": 0, "filler": 99}


def test_build_cluster_feature_table_single_row(mock_vocab):
    """
    Ensures that an MSA with only one sequence skips clustering
    and returns a default variant index.
    """
    # 1. Arrange: Create a single-row DataFrame
    df = pd.DataFrame([["M", "A", "Y"]], index=["genome_alpha"], columns=["pos1", "pos2", "pos3"])
    genomes = ["genome_alpha"]
    cols = ["pos1", "pos2", "pos3"]

    # 2. Act: Call the function with a high cluster request (e.g., 16)
    df_clusters, df_cluster_info = build_cluster_feature_table(
        df=df,
        genomes=genomes,
        columns_subset=cols,
        num_clusters=16,
        matrix="identity",
        vocab_dict=mock_vocab,
    )

    # 3. Assert: Verify the short-circuit behavior

    # Check that the index matches the encoded token of "1"
    # Based on mock_vocab, "1" maps to 101
    expected_index = "101"
    assert df_clusters.index[0] == expected_index

    # Check that data integrity is maintained
    assert df_clusters.iloc[0].tolist() == ["M", "A", "Y"]

    # Check metadata reflects the single row status
    assert df_cluster_info["num_clusters"].iloc[0] == 1
    assert df_cluster_info["clustering_algorithm"].iloc[0] == "single_row_default"

    # Verify we still have the original row count
    assert len(df_clusters) == 1


def test_build_cluster_feature_table_identical_sequences(mock_vocab):
    data = [["A", "C", "T"], ["A", "C", "T"], ["A", "C", "T"]]
    df = pd.DataFrame(data, index=["G1", "G2", "G3"], columns=["p1", "p2", "p3"])

    df_clusters, _ = build_cluster_feature_table(
        df, ["G1", "G2", "G3"], ["p1", "p2", "p3"], 16, "identity", mock_vocab
    )

    # SUCCESS 1: Ensure we didn't lose rows
    assert len(df_clusters) == 3
    # SUCCESS 2: All identical rows get the same variant label
    assert all(idx == "101" for idx in df_clusters.index)


def test_build_cluster_feature_table_insufficient_diversity(mock_vocab):
    # Two unique patterns: [A, C, T] and [G, G, G]
    data = [["A", "C", "T"], ["A", "C", "T"], ["G", "G", "G"]]
    df = pd.DataFrame(data, index=["G1", "G2", "G3"], columns=["p1", "p2", "p3"])

    df_clusters, df_info = build_cluster_feature_table(
        df, ["G1", "G2", "G3"], ["p1", "p2", "p3"], 16, "identity", mock_vocab
    )

    # SUCCESS 3: num_clusters is capped at 2
    assert df_info["num_clusters"].iloc[0] == 2
    # SUCCESS 4: Index contains encoded tokens for '1' or '2' (101 or 102)
    assert set(df_clusters.index).issubset({"101", "102"})


def test_cluster_sequences_fit_to_budget():
    """
    Test that the function correctly forces a specific number of clusters
    when the number of unique variants (found at 1e-7) exceeds n_clusters.
    """
    # 1. Create a 4x4 distance matrix representing 4 distinct sequences.
    # Distances are all > 1e-7, so the first pass will find 4 unique variants.
    # We'll set it up so sequences [0,1] are closer to each other than to [2,3].
    dist_matrix = np.array(
        [[0.0, 0.2, 0.8, 0.9], [0.2, 0.0, 0.7, 0.8], [0.8, 0.7, 0.0, 0.1], [0.9, 0.8, 0.1, 0.0]]
    )

    # 2. Set n_clusters to 2.
    # Pass 1: Finds 4 variants (since 0.1 and 0.2 are > 1e-7).
    # Pass 2: Should trigger because 4 > 2, forcing them into 2 clusters.
    n_clusters = 2
    results = cluster_sequences_agglomerate(dist_matrix, n_clusters=n_clusters)

    # 3. Assertions
    unique_clusters = set(results)

    # Check that it actually reduced the count to our 'budget'
    assert (
        len(unique_clusters) == n_clusters
    ), f"Expected {n_clusters} clusters, but got {len(unique_clusters)}"

    # Check that the clustering makes biological/mathematical sense
    # Sequence 0 and 1 should be in the same cluster; 2 and 3 in the other.
    assert results[0] == results[1], "Sequence 0 and 1 should be grouped together"
    assert results[2] == results[3], "Sequence 2 and 3 should be grouped together"
    assert results[0] != results[2], "Sequence 0 and 2 should be in different clusters"
