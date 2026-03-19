import numpy as np
import pytest
from variants.distance import BLOSUM_GAP_TO_GAP, calculate_distance

# 1. Setup a "Stress-Test" MSA
# Seq 0: Reference
# Seq 1: Identical to 0 but with 80% gaps (Tests Gap-Neutralization)
# Seq 2: Totally different amino acids (Tests Diversity)
# Seq 3: Single residue (Tests Edge-case normalization)
MOCK_MSA = ["MFAV---GLLDAIR", "MFAV----------", "KLRW---VVMETLS", "M-------------"]


@pytest.mark.parametrize("matrix_type", ["identity", "blosum62"])
def test_distance_engine_parity(matrix_type):
    """
    Contract Test: Ensures Symmetric (Training) and Asymmetric (Inference)
    engines produce bit-identical results for complex alignments.
    """

    # A. Simulate Training (Symmetric NxN)
    # This uses the same list for both dimensions
    training_mtx = calculate_distance(MOCK_MSA, MOCK_MSA, matrix=matrix_type)

    # B. Simulate Inference (Asymmetric MxN)
    # We take the first sequence as a 'query' and compare it to the full gallery
    query = [MOCK_MSA[0]]
    inference_mtx = calculate_distance(query, MOCK_MSA, matrix=matrix_type)

    # --- Assertion 1: Diagonal Integrity ---
    # Every sequence must have a 0.0 distance to itself
    for i in range(len(MOCK_MSA)):
        assert training_mtx[i, i] == pytest.approx(
            0.0
        ), f"Self-distance failure for {matrix_type} at index {i}"

    # --- Assertion 2: Training vs Inference Bit-Parity ---
    # The first row of the NxN training matrix must match the 1xN inference matrix
    np.testing.assert_allclose(
        training_mtx[0],
        inference_mtx[0],
        atol=1e-7,
        err_msg=f"Engine drift detected in {matrix_type}: Training row 0 != Inference query",
    )


@pytest.mark.parametrize("matrix_type, expected_floor", [("identity", 0.05), ("blosum62", 0.10)])
def test_gap_neutralization_effectiveness(matrix_type, expected_floor):
    """
    Ensures that shared gaps do not bloat distance to 1.0 (The Radius-1 bug fix).
    """
    # Seq 0 and Seq 1 are identical in their overlapping residues ("MFAV")
    # but Seq 1 is missing the rest of the protein.
    dist_mtx = calculate_distance([MOCK_MSA[0]], [MOCK_MSA[1]], matrix=matrix_type)
    dist = dist_mtx[0, 0]

    # With Gap-Neutralization (BLOSUM) and Pairwise Deletion (Identity),
    # this distance should be 0.0.
    # (Note: In your clustering logic, this 0.0 will trigger the 0.1 fallback/floor)
    assert dist < expected_floor, f"Gap-to-Gap penalty is leaking into {matrix_type}. Dist: {dist}"


def test_blosum_config_sync():
    """
    Safety check for the global configuration.
    If this fails, someone changed the constant without updating the architecture.
    """
    assert (
        BLOSUM_GAP_TO_GAP == 0.0
    ), "BLOSUM_GAP_TO_GAP is not 0.0. This will cause Radius = 1.0 bloat in gappy shards (alignments with many '-')."
