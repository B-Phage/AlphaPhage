from typing import cast

import numpy as np
from Bio.Align import substitution_matrices


def calculate_distance_matrix(alignment: list[str] | np.ndarray, matrix: str | dict) -> np.ndarray:
    """
    Calculate a symmetric distance matrix for a set of aligned sequences.

    Parameters
    ----------
    alignment : list of str or np.ndarray
        Multiple sequence alignment (rows = sequences, columns = positions).
    matrix : str or dict
        Distance type. Options:
        - "identity" : 1 - fraction of identical residues
        - "blosum62" : normalized BLOSUM62 similarity
        - dict : custom dict-based substitution scores

    Returns
    -------
    np.ndarray
        Symmetric distance matrix (shape = [num_sequences, num_sequences]).
    """
    if isinstance(matrix, dict):
        return _distance_matrix_dict(alignment, matrix)
    elif matrix == "identity":
        return _distance_matrix_identity(alignment)
    elif matrix == "blosum62":
        return _distance_matrix_blosum62(alignment)
    else:
        raise ValueError(
            "Invalid distance matrix type. Choose 'identity', 'blosum62', or provide a dict."
        )


# -------------------------
# Internal helper functions
# -------------------------


def _distance_matrix_identity(alignment: list[str] | np.ndarray) -> np.ndarray:
    """Distance = 1 - fraction of identical residues"""
    n = len(alignment)
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            seq1, seq2 = alignment[i], alignment[j]
            distance = 1 - sum(c1 == c2 for c1, c2 in zip(seq1, seq2)) / len(seq1)
            mat[i, j] = mat[j, i] = distance
    return mat


def _distance_matrix_dict(alignment: list[str] | np.ndarray, score_dict: dict) -> np.ndarray:
    """Distance = negative normalized score from a custom dictionary"""
    n = len(alignment)
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            seq1, seq2 = alignment[i], alignment[j]
            score = sum(score_dict[c1][c2] for c1, c2 in zip(seq1, seq2))
            distance = -score / len(seq1)
            mat[i, j] = mat[j, i] = distance
    return mat


def _distance_matrix_blosum62(alignment: list[str] | np.ndarray) -> np.ndarray:
    """Distance based on BLOSUM62, normalized to [0,1]"""
    n = len(alignment)
    mat = np.zeros((n, n))
    blosum62 = cast(dict, substitution_matrices.load("BLOSUM62"))
    GAP_PENALTY = -4
    MAX_SCORE = max(blosum62.values())

    for i in range(n):
        for j in range(i + 1, n):
            seq1, seq2 = alignment[i], alignment[j]
            score = 0
            for c1, c2 in zip(seq1, seq2):
                if c1 == "-" or c2 == "-":
                    score += GAP_PENALTY
                else:
                    pair_score = blosum62.get((c1, c2))
                    if pair_score is None:
                        pair_score = blosum62.get((c2, c1), 0)
                    score += int(pair_score)
            distance = 1 - score / (len(seq1) * MAX_SCORE)
            mat[i, j] = mat[j, i] = distance
    return mat
