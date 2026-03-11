import numpy as np
import pytest
from variants.distance import calculate_distance_matrix

# -------------------------
# Test Identity Calculation
# -------------------------


def test_identity_identical_sequences():
    """Identical sequences must have 0.0 distance."""
    alignment = ["ACGT", "ACGT"]
    matrix = calculate_distance_matrix(alignment, "identity")

    expected = np.array([[0.0, 0.0], [0.0, 0.0]])
    np.testing.assert_allclose(matrix, expected, atol=1e-7)


def test_identity_completely_different():
    """Sequences with 0% identity should have 1.0 distance."""
    alignment = ["AAAA", "TTTT"]
    matrix = calculate_distance_matrix(alignment, "identity")

    # Pair (0,1) should be 1.0
    assert matrix[0, 1] == 1.0
    assert matrix[1, 0] == 1.0


def test_identity_partial_match():
    """Check math for 50% identity."""
    alignment = ["AACC", "AAGG"]
    matrix = calculate_distance_matrix(alignment, "identity")

    # 2 matches out of 4 = 0.5 identity -> 0.5 distance
    assert matrix[0, 1] == 0.5


# -------------------------
# Test BLOSUM62 Calculation
# -------------------------


def test_blosum62_identical_normalization():
    """Verify that identical sequences result in 0.0 distance regardless of complexity."""
    # Simple AA vs Complex AA
    alignment = ["AAAAA", "AAAAA", "WWWWW"]
    matrix = calculate_distance_matrix(alignment, "blosum62")

    # Diagonal must be 0
    assert matrix[0, 0] == 0.0
    assert matrix[2, 2] == 0.0
    # Identical A's must be 0.0 even though their raw score is lower than W's
    assert matrix[0, 1] == 0.0


def test_blosum62_functional_sensitivity():
    """BLOSUM should show smaller distance for conservative mutations."""
    # L -> I is conservative (both hydrophobic)
    # L -> P is radical (Proline breaks structure)
    alignment = ["LLLL", "IIII", "PPPP"]
    matrix = calculate_distance_matrix(alignment, "blosum62")

    dist_conservative = matrix[0, 1]  # L vs I
    dist_radical = matrix[0, 2]  # L vs P

    assert dist_conservative < dist_radical
    print(f"Conservative: {dist_conservative:.4f}, Radical: {dist_radical:.4f}")


# -------------------------
# Test Custom Dict Logic
# -------------------------


def test_custom_dict_logic():
    score_dict = {"A": {"A": 10, "T": 0}, "T": {"A": 0, "T": 10}}
    alignment = ["AA", "AT"]
    # Score for AA vs AT = 10 + 0 = 10. Length = 2.
    # Distance = -10 / 2 = -5.0
    matrix = calculate_distance_matrix(alignment, score_dict)
    assert matrix[0, 1] == -5.0


# -------------------------
# Test Edge Cases & Shapes
# -------------------------


def test_empty_alignment():
    """Test behavior with zero sequences."""
    matrix = calculate_distance_matrix([], "identity")
    assert matrix.shape == (0, 0)


def test_single_sequence():
    """A single sequence should result in a 1x1 matrix of [0.0]."""
    matrix = calculate_distance_matrix(["ACGT"], "identity")
    assert matrix.shape == (1, 1)
    assert matrix[0, 0] == 0.0


def test_numpy_input_compatibility():
    """Ensure function accepts numpy array as well as list of strings."""
    aln_list = ["AAA", "AAT"]
    aln_np = np.array([list("AAA"), list("AAT")])

    res_list = calculate_distance_matrix(aln_list, "identity")
    res_np = calculate_distance_matrix(aln_np, "identity")

    np.testing.assert_array_equal(res_list, res_np)


def test_symmetry():
    """The matrix must be perfectly symmetric."""
    alignment = ["ACGT", "TGCA", "ACGA", "GGGG"]
    matrix = calculate_distance_matrix(alignment, "blosum62")
    np.testing.assert_allclose(matrix, matrix.T, atol=1e-12)


def test_invalid_matrix_type():
    with pytest.raises(ValueError, match="Invalid distance matrix type"):
        calculate_distance_matrix(["AAA"], "invalid_option")


def test_blosum62_empty_alignment():
    """
    Test the specific case where len(alignment) == 0.
    Should return a 0x0 matrix rather than raising an error.
    """
    from variants.distance import _distance_matrix_blosum62_vectorized

    # Test with empty list
    matrix_list = _distance_matrix_blosum62_vectorized([])
    assert isinstance(matrix_list, np.ndarray)
    assert matrix_list.shape == (0, 0)
    assert matrix_list.dtype == np.float64

    # Test with empty numpy array
    empty_np = np.array([]).reshape(0, 0)
    matrix_np = _distance_matrix_blosum62_vectorized(empty_np)
    assert matrix_np.shape == (0, 0)


def test_blosum62_single_sequence():
    """
    Test the case with exactly one sequence.
    Should return a 1x1 matrix containing [0.0].
    """
    from variants.distance import _distance_matrix_blosum62_vectorized

    alignment = ["ACDEFGHIKLMNPQRSTVWY"]
    matrix = _distance_matrix_blosum62_vectorized(alignment)

    assert matrix.shape == (1, 1)
    # Distance to itself must be exactly 0.0
    assert matrix[0, 0] == 0.0
