from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from variants.inference_mapper import InferenceMapper

# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def mapper():
    """Returns an InferenceMapper with a standard 1.1 buffer."""
    return InferenceMapper(buffer_factor=1.1)


@pytest.fixture
def gallery_df():
    """Mock gallery representing a Parquet Shard."""
    return pd.DataFrame(
        {"variant_id": ["V1", "V2"], "sequence_aligned": ["AAAA", "CCCC"], "radius": [0.1, 0.2]}
    )


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------


def test_map_to_variants_exact_match(mapper, gallery_df):
    """Verifies that sequences in the shortcut map skip distance calculation."""
    queries_df = pd.DataFrame(
        {
            "organism_id": ["org_exact"],
            "hmm_id": ["H1"],
            "sequence_aligned": ["AAAA"],  # Matches V1 in gallery
        }
    )

    with patch("variants.inference_mapper.calculate_distance") as mock_dist:
        result = mapper.map_to_variants(queries_df, gallery_df, "identity")

        assert result.iloc[0]["assigned_variant_id"] == "V1"
        assert result.iloc[0]["mapping_method"] == "exact_match"
        assert result.iloc[0]["distance"] == 0.0
        # Ensure heavy math was NEVER called
        mock_dist.assert_not_called()


@patch("variants.inference_mapper.calculate_distance")
def test_map_to_variants_within_radius(mock_dist, mapper, gallery_df):
    """Verifies assignment when a sequence is a 'Miss' but within the radius buffer."""
    queries_df = pd.DataFrame(
        {"organism_id": ["org_near"], "hmm_id": ["H1"], "sequence_aligned": ["AAAT"]}  # Cache miss
    )

    # Distance to V1 (0.1 radius) is 0.105.
    # Threshold = 0.1 * 1.1 = 0.11.
    # 0.105 < 0.11 -> Should assign V1.
    mock_dist.return_value = np.array([[0.105, 0.5]])

    result = mapper.map_to_variants(queries_df, gallery_df, "identity")

    assert result.iloc[0]["assigned_variant_id"] == "V1"
    assert result.iloc[0]["mapping_method"] == "distance_identity"
    assert result.iloc[0]["distance"] == 0.105


@patch("variants.inference_mapper.calculate_distance")
def test_map_to_variants_outside_radius_unknown(mock_dist, mapper, gallery_df):
    """Verifies 'UNKNOWN_NEW_VARIANT' assignment when distance exceeds radius * buffer."""
    queries_df = pd.DataFrame(
        {"organism_id": ["org_far"], "hmm_id": ["H1"], "sequence_aligned": ["GGGG"]}  # Cache miss
    )

    # Best match is V2 (0.2 radius). Dist is 0.3.
    # Threshold = 0.2 * 1.1 = 0.22.
    # 0.3 > 0.22 -> Should be UNKNOWN.
    mock_dist.return_value = np.array([[0.8, 0.3]])

    result = mapper.map_to_variants(queries_df, gallery_df, "blosum62")

    assert result.iloc[0]["assigned_variant_id"] == "UNKNOWN_NEW_VARIANT"
    assert result.iloc[0]["distance"] == 0.3
    assert result.iloc[0]["mapping_method"] == "distance_blosum62"


@patch("variants.inference_mapper.calculate_distance")
def test_map_to_variants_mixed_batch(mock_dist, mapper, gallery_df):
    """Tests a batch containing both exact matches and distance calculations."""
    queries_df = pd.DataFrame(
        {
            "organism_id": ["org1", "org2"],
            "hmm_id": ["H1", "H1"],
            "sequence_aligned": ["AAAA", "GGGG"],  # org1: Match, org2: Miss
        }
    )

    # Mock distance for the single miss (org2)
    mock_dist.return_value = np.array([[0.5, 0.5]])

    result = mapper.map_to_variants(queries_df, gallery_df, "identity")

    assert len(result) == 2
    # Check order preservation/results
    assert result.query("organism_id == 'org1'")["mapping_method"].iloc[0] == "exact_match"
    assert result.query("organism_id == 'org2'")["mapping_method"].iloc[0] == "distance_identity"


def test_map_to_variants_empty_queries(mapper, gallery_df):
    """Ensures graceful handling of empty query DataFrames."""
    empty_df = pd.DataFrame(columns=["organism_id", "hmm_id", "sequence_aligned"])

    result = mapper.map_to_variants(empty_df, gallery_df, "identity")

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0


@patch("variants.inference_mapper.calculate_distance")
def test_vectorization_input_types(mock_dist, mapper, gallery_df):
    """Ensures sequences are correctly converted to 2D byte arrays for the engine."""
    queries_df = pd.DataFrame(
        {"organism_id": ["org1"], "hmm_id": ["H1"], "sequence_aligned": ["ABC"]}
    )

    # We don't care about the result, just the call arguments
    mock_dist.return_value = np.array([[0.0, 0.0]])

    mapper.map_to_variants(queries_df, gallery_df, "identity")

    args, kwargs = mock_dist.call_args

    # Verify q_matrix and target are numpy arrays with dtype |S1
    assert isinstance(kwargs["query"], np.ndarray)
    assert kwargs["query"].dtype == "|S1"
    assert kwargs["target"].dtype == "|S1"


def test_inference_mapper_custom_config(gallery_df):
    """Verifies that custom buffer and consensus_ratio are applied."""
    # Use a very strict consensus_ratio (1.0) - no override allowed
    mapper = InferenceMapper(buffer_factor=1.5, consensus_ratio=1.0)

    queries_df = pd.DataFrame(
        {"organism_id": ["org1"], "hmm_id": ["H1"], "sequence_aligned": ["AAAT"]}
    )

    with (
        patch("variants.inference_mapper.calculate_distance") as mock_dist,
        patch("variants.inference_mapper.rigorous_assignment") as mock_assign,
    ):

        mock_dist.return_value = np.array([[0.1, 0.15]])
        mock_assign.return_value = (["V1"], [0.1])

        mapper.map_to_variants(queries_df, gallery_df, "identity")

        # Check that thresholds passed to rigorous_assignment were
        # multiplied by the custom buffer_factor (1.5)
        args, kwargs = mock_assign.call_args
        passed_thresholds = kwargs["cluster_thresholds"]

        # Original V1 radius was 0.1, now should be 0.15
        assert passed_thresholds["V1"] == pytest.approx(0.15)
        # Check consensus_ratio was passed
        assert kwargs["consensus_ratio"] == 1.0


def test_inference_mapper_exact_match_priority(mapper, gallery_df):
    """Ensure that exact matches are prioritized over even a 0.0 distance matrix result."""
    queries_df = pd.DataFrame(
        {
            "organism_id": ["org1"],
            "hmm_id": ["H1"],
            "sequence_aligned": ["AAAA"],  # Exact match for V1
        }
    )

    # Even if we "accidentally" pass it to the engine, shortcut must win
    with patch("variants.inference_mapper.calculate_distance") as mock_dist:
        results = mapper.map_to_variants(queries_df, gallery_df, "identity")
        assert results.iloc[0]["mapping_method"] == "exact_match"
        mock_dist.assert_not_called()
