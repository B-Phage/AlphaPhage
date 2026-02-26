import numpy as np
import pytest
from Bio.Align import substitution_matrices
from variants.distance import (
    _distance_matrix_blosum62,
    _distance_matrix_dict,
    _distance_matrix_identity_vectorized,
    calculate_distance_matrix,
)

# -------------------------
# Sample alignments
# -------------------------

ALIGNMENT_SIMPLE = ["AAA", "AAT", "ATT"]
ALIGNMENT_WITH_GAPS = ["A-A", "AAA", "T-A"]

CUSTOM_DICT = {"A": {"A": 1, "T": 0}, "T": {"A": 0, "T": 1}, "-": {"A": 0, "T": 0, "-": 0}}


# -------------------------
# Test identity distance
# -------------------------


def test_identity_distance_matrix_symmetry():
    mat = _distance_matrix_identity_vectorized(ALIGNMENT_SIMPLE)
    assert mat.shape == (3, 3)
    # Diagonal should be zero
    np.testing.assert_array_equal(np.diag(mat), np.zeros(3))
    # Symmetry
    assert np.allclose(mat, mat.T)


def test_identity_distance_values():
    mat = _distance_matrix_identity_vectorized(["AAA", "AAT"])
    expected = np.array([[0.0, 1 - 2 / 3], [1 - 2 / 3, 0.0]])
    np.testing.assert_allclose(mat, expected)


# -------------------------
# Test custom dictionary distance
# -------------------------


def test_dict_distance_matrix_symmetry():
    mat = _distance_matrix_dict(ALIGNMENT_SIMPLE, CUSTOM_DICT)
    assert mat.shape == (3, 3)
    assert np.allclose(mat, mat.T)
    # Diagonal should be zero
    np.testing.assert_array_equal(np.diag(mat), np.zeros(3))


def test_dict_distance_values():
    mat = _distance_matrix_dict(["AA", "AT"], CUSTOM_DICT)
    # Score = sum of pairwise scores, distance = -score / length
    # seq1=AA, seq2=AT -> scores: 1,0 -> sum=1 -> distance=-1/2=-0.5
    expected = np.array([[0.0, -0.5], [-0.5, 0.0]])
    np.testing.assert_allclose(mat, expected)


# -------------------------
# Test BLOSUM62 distance
# -------------------------


def test_blosum62_distance_matrix_shape_symmetry():
    mat = _distance_matrix_blosum62(ALIGNMENT_SIMPLE)
    assert mat.shape == (3, 3)
    assert np.allclose(mat, mat.T)
    np.testing.assert_array_equal(np.diag(mat), np.zeros(3))


def test_blosum62_distance_known_values():
    # Simple check: distance between identical sequences should be 1 - max_score/ (len*MAX_SCORE) = 0
    mat = _distance_matrix_blosum62(["AAA", "AAA"])
    score = 4 + 4 + 4  # BLOSUM62 scores for A:A pairs
    MAX_SCORE = 11  # this matches the function
    expected_distance = 1 - score / (3 * MAX_SCORE)
    expected = np.array([[0.0, expected_distance], [expected_distance, 0.0]])
    np.testing.assert_allclose(mat, expected)


def test_blosum62_with_gaps():
    mat = _distance_matrix_blosum62(ALIGNMENT_WITH_GAPS)
    blosum62 = substitution_matrices.load("BLOSUM62")
    MAX_SCORE = max(blosum62.values())
    GAP_PENALTY = -4
    seq1, seq2 = ALIGNMENT_WITH_GAPS[0], ALIGNMENT_WITH_GAPS[1]
    score = 0
    for c1, c2 in zip(seq1, seq2):
        if c1 == "-" or c2 == "-":
            score += GAP_PENALTY
        else:
            score += blosum62.get((c1, c2), blosum62.get((c2, c1), 0))
    expected_distance = 1 - score / (len(seq1) * MAX_SCORE)
    assert np.isclose(mat[0, 1], expected_distance)


def test_blosum62_reverse_lookup_branch(monkeypatch):
    """
    Ensure branch is covered where (c1, c2) is missing
    but (c2, c1) exists in the substitution matrix.
    """

    # Fake asymmetric BLOSUM matrix
    fake_blosum = {
        ("A", "R"): 5  # Only forward direction exists
        # ('R','A') intentionally missing
    }

    # Monkeypatch substitution_matrices.load to return fake matrix
    from Bio.Align import substitution_matrices

    monkeypatch.setattr(substitution_matrices, "load", lambda _: fake_blosum)

    # Sequences arranged so first lookup fails:
    # ('R','A') not in dict
    # fallback to ('A','R') works
    alignment = ["RA", "AA"]

    mat = _distance_matrix_blosum62(alignment)

    # Score calculation:
    # Position 1: R vs A → fallback → 5
    # Position 2: A vs A → not in dict → fallback 0
    score = 5
    MAX_SCORE = max(fake_blosum.values())  # = 5
    expected_distance = 1 - score / (2 * MAX_SCORE)

    assert np.isclose(mat[0, 1], expected_distance)


# -------------------------
# Test main function
# -------------------------


def test_calculate_distance_matrix_identity():
    mat1 = calculate_distance_matrix(ALIGNMENT_SIMPLE, "identity")
    mat2 = _distance_matrix_identity_vectorized(ALIGNMENT_SIMPLE)
    np.testing.assert_allclose(mat1, mat2)


def test_calculate_distance_matrix_dict():
    mat1 = calculate_distance_matrix(ALIGNMENT_SIMPLE, CUSTOM_DICT)
    mat2 = _distance_matrix_dict(ALIGNMENT_SIMPLE, CUSTOM_DICT)
    np.testing.assert_allclose(mat1, mat2)


def test_calculate_distance_matrix_blosum62():
    mat1 = calculate_distance_matrix(ALIGNMENT_SIMPLE, "blosum62")
    mat2 = _distance_matrix_blosum62(ALIGNMENT_SIMPLE)
    np.testing.assert_allclose(mat1, mat2)


def test_calculate_distance_matrix_invalid_type():
    with pytest.raises(ValueError):
        calculate_distance_matrix(ALIGNMENT_SIMPLE, "invalid_type")


# -------------------------
# Edge case tests
# -------------------------


def test_empty_alignment():
    mat = calculate_distance_matrix([], "identity")
    np.testing.assert_array_equal(mat, np.zeros((0, 0)))


def test_single_sequence():
    mat = calculate_distance_matrix(["AAA"], "identity")
    np.testing.assert_array_equal(mat, np.zeros((1, 1)))
