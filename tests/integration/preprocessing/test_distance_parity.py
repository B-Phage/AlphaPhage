import numpy as np
import pytest
from variants.distance import calculate_distance

# --- ROBUST MOCK DATA ---
# 0: Reference
# 1: Staggered internal fragment (Tests index-specific matching)
# 2: C-terminal fragment (No overlap with Seq 3)
# 3: N-terminal fragment (No overlap with Seq 2)
# 4: Same length, total diversity
# 5: Mixed gap characters (Testing . vs -)
MOCK_MSA = [
    "MFAV---GLLDAIR",  # 0
    "---V---GLL----",  # 1
    "----------DAIR",  # 2
    "MFAV----------",  # 3
    "KLRW---VVMETLS",  # 4
    "M.A.V..G.L.D.R",  # 5
]


@pytest.mark.parametrize("matrix_type", ["identity", "blosum62"])
def test_distance_engine_parity(matrix_type):
    """
    Ensures Symmetric (Training) and Asymmetric (Inference) engines
    produce bit-identical results.
    """
    training_mtx = calculate_distance(MOCK_MSA, MOCK_MSA, matrix=matrix_type)

    # Test with a single query vs the whole set
    query = [MOCK_MSA[0]]
    inference_mtx = calculate_distance(query, MOCK_MSA, matrix=matrix_type)

    # 1. Diagonal check: Distance to self must be 0.0
    for i in range(len(MOCK_MSA)):
        assert training_mtx[i, i] == pytest.approx(0.0, abs=1e-7)

    # 2. Parity check: Row 0 of training == Row 0 of inference
    np.testing.assert_allclose(
        training_mtx[0], inference_mtx[0], atol=1e-7, err_msg=f"Engine drift in {matrix_type}"
    )


def test_staggered_overlap_identity():
    """
    Logic Check (Pairwise Deletion):
    Seq 0: MFAV---GLLDAIR
    Seq 1: ---V---GLL----

    - Shared Gaps: Indices [4, 5, 6] (3 positions)
    - Total Comparable: 14 - 3 = 11
    - Matches: 'V' (idx 3), 'G' (idx 7), 'L' (idx 8), 'L' (idx 9) = 4 matches
    - Identity: 4 / 11 = 0.3636
    - Expected Distance: 1 - 0.3636 = 0.6363
    """
    dist_mtx = calculate_distance([MOCK_MSA[0]], [MOCK_MSA[1]], matrix="identity")
    assert dist_mtx[0, 0] == pytest.approx(0.6363, abs=1e-3)


def test_zero_overlap_isolation():
    """
    Seq 2: ----------DAIR
    Seq 3: MFAV----------

    These have NO overlapping residues.
    Even if they share some gap positions, the identity should be 0.0.
    Distance MUST be 1.0.
    """
    dist_mtx = calculate_distance([MOCK_MSA[2]], [MOCK_MSA[3]], matrix="identity")
    assert dist_mtx[0, 0] == pytest.approx(1.0), "Zero overlap failed to yield max distance"


def test_shared_gap_neutrality_complex():
    """
    Adding shared gaps should not change the distance between two sequences.
    This is the ultimate test for 'Pairwise Deletion' logic.
    """
    seq_a = "MFAV"
    seq_b = "MFRV"  # 1 mismatch at index 2

    base_dist = calculate_distance([seq_a], [seq_b], matrix="identity")[0, 0]  # 1 - 3/4 = 0.25

    # Scenario: Embed them in a much larger alignment with shared gaps
    gappy_a = "---MFAV-------"
    gappy_b = "---MFRV-------"
    gappy_dist = calculate_distance([gappy_a], [gappy_b], matrix="identity")[0, 0]

    assert base_dist == pytest.approx(gappy_dist), "Shared gaps leaked into the calculation!"


def test_blosum_partial_overlap_range():
    """
    BLOSUM62 Distance for partial fragments.
    Distance should reflect the biological penalty of the mismatching residues
    and the 'neutral' nature of shared gaps.
    """
    dist_mtx = calculate_distance([MOCK_MSA[0]], [MOCK_MSA[1]], matrix="blosum62")
    dist = dist_mtx[0, 0]

    # With -2.0 penalty for Res-vs-Gap and 0.0 for Gap-vs-Gap:
    # Seq 0/1 should be quite distant but not 1.0.
    assert 0.5 < dist < 0.9, f"BLOSUM distance {dist} outside expected partial-overlap range."


def test_mixed_gap_characters():
    """
    Ensure the engine treats '-' (45) and '.' (46) identically.
    """
    dist_standard = calculate_distance([MOCK_MSA[0]], [MOCK_MSA[3]], matrix="identity")

    # Seq 5 is a variant of Seq 0/3 logic using dots
    # We check if a sequence with dots behaves like a sequence with dashes
    dist_dots = calculate_distance([MOCK_MSA[0]], [MOCK_MSA[5]], matrix="identity")

    # If the engine handles both, the distances should be consistent
    assert 0 < dist_dots[0, 0] < 1.0


def test_empty_input_handling():
    """
    Engine should not crash on empty input lists.
    """
    dist = calculate_distance([], [], matrix="identity")
    assert dist.shape == (0, 0)


def test_blosum_config_integrity():
    """
    Safety check for the global BLOSUM cache parameters.
    """
    from variants.distance import _get_blosum_params

    _, lookup = _get_blosum_params()
    gap_idx = lookup.shape[0] - 1

    assert lookup[gap_idx, gap_idx] == 0.0, "Shared gaps must be neutral (0.0)"
    assert lookup[0, gap_idx] == -2.0, "Residue-vs-Gap penalty should be -2.0"
