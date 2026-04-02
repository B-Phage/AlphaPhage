from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from variants.inference_mapper import InferenceMapper  # Replace with actual module name


@pytest.fixture
def mapper():
    return InferenceMapper(buffer_factor=1.1, consensus_ratio=1.2, fallback_radius=0.5)


@pytest.fixture
def sample_gallery():
    return pd.DataFrame(
        {
            "sequence_aligned": ["AAAA", "BBBB"],
            "variant_id": ["V1", "V2"],
            "radius": [1.0, 2.0],
            "organism_id": ["ORG1", "ORG2"],
        }
    )


@pytest.fixture
def sample_queries():
    return pd.DataFrame(
        {
            "sequence_aligned": ["AAAA", "CCCC"],  # AAAA is exact match, CCCC is a miss
            "organism_id": ["Q1", "Q2"],
        }
    )


def test_map_to_variants_full_flow(mapper, sample_gallery, sample_queries):
    with (
        patch("variants.inference_mapper.calculate_distance") as mock_dist,
        patch("variants.inference_mapper.rigorous_assignment") as mock_assign,
    ):

        # Ensure the distances match exactly what the class expects to process
        mock_dist.return_value = np.array([[0.5, 0.8]])

        # We expect the 'distance' column to contain the score from the second array
        mock_assign.return_value = (np.array(["V1"]), np.array([0.5], dtype=np.float64))

        result_df = mapper.map_to_variants(
            queries_df=sample_queries,
            gallery_df=sample_gallery,
            matrix_type="blosum62",
            hmm_id="HMM_001",
        )

        dist_match = result_df[result_df["mapping_method"] == "distance_blosum62"]

        # Add a small tolerance if using floats, though 0.5 should be exact
        assert dist_match.iloc[0]["variant_id"] == "V1"
        assert float(dist_match.iloc[0]["distance"]) == 0.5


def test_map_to_variants_all_exact_matches(mapper, sample_gallery):
    """Tests the path where no distance calculation is needed (100% hits)."""
    queries = pd.DataFrame({"sequence_aligned": ["AAAA", "BBBB"], "organism_id": ["Q1", "Q2"]})

    with patch("variants.distance.calculate_distance") as mock_dist:
        result_df = mapper.map_to_variants(queries, sample_gallery, "identity", "HMM_001")

        assert len(result_df) == 2
        assert all(result_df["mapping_method"] == "exact_match")
        # Ensure the distance calculation block was skipped
        mock_dist.assert_not_called()


def test_map_to_variants_empty_input(mapper, sample_gallery):
    """Tests behavior with an empty queries DataFrame."""
    empty_queries = pd.DataFrame(columns=["sequence_aligned", "organism_id"])

    result_df = mapper.map_to_variants(empty_queries, sample_gallery, "identity", "HMM_001")

    assert isinstance(result_df, pd.DataFrame)
    assert len(result_df) == 0


def test_initialization_params():
    """Ensure custom init parameters are assigned correctly."""
    custom_mapper = InferenceMapper(buffer_factor=2.0, consensus_ratio=1.5, fallback_radius=0.1)
    assert custom_mapper.buffer == 2.0
    assert custom_mapper.consensus_ratio == 1.5
    assert custom_mapper.fallback_radius == 0.1
