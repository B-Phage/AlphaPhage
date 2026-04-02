import numpy as np
import pytest
from variants.distance import (
    _engine_blosum62,
    _get_blosum_params,
    _to_byte_array,
    calculate_distance,
    calculate_distance_matrix,
)

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
    with pytest.raises(ValueError, match="Unsupported matrix: invalid_option"):
        calculate_distance_matrix(["AAA"], "invalid_option")


def test_blosum62_empty_alignment():
    """
    Test the specific case where len(alignment) == 0.
    Should return a 0x0 matrix rather than raising an error or 1D array.
    """
    c2i, lookup = _get_blosum_params()

    # 1. Test the internal engine directly (Internal Logic Check)
    # The engine expects 2D byte arrays from _to_byte_array
    empty_q = np.zeros((0, 0), dtype="|S1")
    empty_t = np.zeros((0, 0), dtype="|S1")

    matrix_engine = _engine_blosum62(empty_q, empty_t, c2i, lookup)

    assert isinstance(matrix_engine, np.ndarray)
    assert matrix_engine.shape == (0, 0)
    assert matrix_engine.dtype == np.float64

    # 2. Test via public API with empty list (End-to-End Check)
    # This proves _to_byte_array and the engine work together
    matrix_list = calculate_distance([], [], matrix="blosum62")
    assert matrix_list.shape == (0, 0)

    # 3. Test with empty numpy array reshaped to 2D
    # This simulates a 0-sequence alignment with 0-length positions
    empty_np = np.array([], dtype="|S1").reshape(0, 0)
    matrix_np = calculate_distance(empty_np, empty_np, matrix="blosum62")
    assert matrix_np.shape == (0, 0)


def test_to_byte_array_empty_input():
    """
    Verify that _to_byte_array enforces 2D shape for empty inputs.
    This prevents the 'IndexError: too many indices' during broadcasting.
    """
    # Empty list should result in (0, 0)
    res_list = _to_byte_array([])
    assert res_list.ndim == 2
    assert res_list.shape == (0, 0)

    # 1D empty array should result in (0, 0)
    res_1d = _to_byte_array(np.array([], dtype="|S1"))
    assert res_1d.ndim == 2
    assert res_1d.shape == (0, 0)


def test_blosum62_single_sequence():
    """
    Test the case with exactly one sequence.
    Should return a 1x1 matrix containing [0.0].
    """
    # Test via the public API (Backwards compatible with your training workflow)
    alignment = ["ACDEFGHIKLMNPQRSTVWY"]
    matrix = calculate_distance(alignment, alignment, matrix="blosum62")

    assert isinstance(matrix, np.ndarray)
    assert matrix.shape == (1, 1)

    # Distance to itself must be exactly 0.0
    # We use approx to handle any infinitesimal floating point jitter
    assert matrix[0, 0] == pytest.approx(0.0)
    assert matrix.dtype == np.float64


def test_blosum62_single_sequence_internal_engine():
    """
    Verify the internal engine handles a 1x1 case correctly
    after being processed by _to_byte_array.
    """
    from variants.distance import _get_blosum_params, _to_byte_array

    c2i, lookup = _get_blosum_params()
    alignment = ["ACDEFGHIKLMNPQRSTVWY"]

    # Simulate the internal pipeline flow
    q_arr = _to_byte_array(alignment)
    t_arr = _to_byte_array(alignment)

    matrix = _engine_blosum62(q_arr, t_arr, c2i, lookup)

    assert matrix.shape == (1, 1)
    assert matrix[0, 0] == 0.0


def test_to_byte_array_1d_numpy_strings():
    """
    Verify that a 1D numpy array of strings (e.g., ['ACT', 'ACG'])
    is correctly expanded into a 2D byte array of characters.
    """
    from variants.distance import _to_byte_array

    # Input: 1D NumPy array of strings
    # This specifically hits: if data.ndim == 1
    input_data = np.array(["ACGT", "TGCA"])

    res = _to_byte_array(input_data)

    # Validation
    assert res.ndim == 2
    assert res.shape == (2, 4)
    assert res.dtype == "|S1"

    # Check specific character mapping
    expected = np.array([["A", "C", "G", "T"], ["T", "G", "C", "A"]], dtype="|S1")

    np.testing.assert_array_equal(res, expected)
