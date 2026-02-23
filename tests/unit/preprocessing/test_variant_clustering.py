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
    return {"0": 0, "1": 1}


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
    vocab_dict,
    mock_distance_matrix,
):
    mock_calc.return_value = mock_distance_matrix

    df_clusters, df_info = build_cluster_feature_table(
        df=simple_alignment_df,
        genomes=list(simple_alignment_df.index),
        columns_subset=["pos1", "pos2", "pos3"],
        num_clusters=2,
        matrix="identity",
        vocab_dict=vocab_dict,
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
    vocab_dict,
    mock_distance_matrix,
):
    mock_calc.return_value = mock_distance_matrix

    df_clusters, df_info = build_cluster_feature_table(
        df=simple_alignment_df,
        genomes=list(simple_alignment_df.index),
        columns_subset=["pos1", "pos2", "pos3"],
        num_clusters=10,  # too many
        matrix="identity",
        vocab_dict=vocab_dict,
    )

    # Should reduce to number of unique samples (2 in this dataset)
    assert df_info.iloc[0]["num_clusters"] <= 3


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
