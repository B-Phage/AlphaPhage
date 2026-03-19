from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from Bio.Align import substitution_matrices

# Global cache for heavy lookup objects
_BLOSUM_CACHE = {}
BLOSUM_GAP_TO_GAP = 0.0  # Neutralize shared gaps


def calculate_distance_matrix(
    alignment: Union[List[str], np.ndarray], matrix: str = "identity"
) -> np.ndarray:
    """
    Backwards compatible wrapper for training.
    Produces a symmetric NxN matrix for a single alignment.
    """
    return calculate_distance(query=alignment, target=alignment, matrix=matrix)


def calculate_distance(
    query: Union[List[str], np.ndarray],
    target: Union[List[str], np.ndarray],
    matrix: str = "identity",
) -> np.ndarray:
    """
    Universal entry point for both Training (Symmetric) and Inference (Asymmetric).

    If query is target, this produces a symmetric NxN matrix.
    If query != target, this produces an MxN matrix.
    """
    # 1. Standardize Inputs to Byte Arrays
    q_arr = _to_byte_array(query)
    t_arr = _to_byte_array(target)

    if matrix == "identity":
        return _engine_identity(q_arr, t_arr)
    elif matrix == "blosum62":
        c2i, lookup = _get_blosum_params()
        return _engine_blosum62(q_arr, t_arr, c2i, lookup)
    else:
        raise ValueError(f"Unsupported matrix: {matrix}")


# --- Core Engines (Shared by Training & Inference) ---


def _engine_identity(q_arr: np.ndarray, t_arr: np.ndarray) -> np.ndarray:
    """Calculates Identity Distance using Pairwise Deletion."""
    # Short-circuit for empty inputs
    if q_arr.size == 0 or t_arr.size == 0:
        return np.zeros((q_arr.shape[0], t_arr.shape[0]), dtype=np.float64)

    # Broadcasting (M, 1, L) == (1, N, L) -> (M, N, L)
    matches = q_arr[:, None, :] == t_arr[None, :, :]

    # ... rest of the shared gaps logic ...
    is_gap_q = (q_arr == 45) | (q_arr == 46)
    is_gap_t = (t_arr == 45) | (t_arr == 46)
    shared_gaps = is_gap_q[:, None, :] & is_gap_t[None, :, :]

    valid_matches = matches & ~shared_gaps
    total_comparable = q_arr.shape[1] - np.sum(shared_gaps, axis=2)
    total_comparable = np.maximum(total_comparable, 1)

    identity = np.sum(valid_matches, axis=2) / total_comparable
    return np.clip(1.0 - identity, 0, 1)


def _engine_blosum62(
    q_arr: np.ndarray, t_arr: np.ndarray, c2i: np.ndarray, lookup: np.ndarray
) -> np.ndarray:
    """Calculates BLOSUM62 Distance using Gap-Neutralization."""
    # Short-circuit for empty inputs
    if q_arr.size == 0 or t_arr.size == 0:
        return np.zeros((q_arr.shape[0], t_arr.shape[0]), dtype=np.float64)
    # Map to integers
    q_int = c2i[q_arr.view("uint8")]
    t_int = c2i[t_arr.view("uint8")]

    M, L = q_int.shape
    N = t_int.shape[0]

    # Calculate Raw Scores
    total_scores = np.zeros((M, N), dtype=np.float64)
    for pos in range(L):
        q_col = q_int[:, pos]
        t_col = t_int[:, pos]
        total_scores += lookup[q_col[:, None], t_col[None, :]]

    # Normalized Distance = 1 - (Score_qt / max(Score_qq, Score_tt))
    # Pre-calculate self-scores for normalization
    # We sum the lookup diagonal for the specific indices in each sequence
    q_self = np.array([lookup[seq, seq].sum() for seq in q_int])
    t_self = np.array([lookup[seq, seq].sum() for seq in t_int])

    max_self = np.maximum.outer(q_self, t_self)
    max_self = np.maximum(max_self, 1.0)

    distance = 1.0 - (total_scores / max_self)
    return np.clip(distance, 0, 1)


# --- Utilities ---


def _to_byte_array(data: Union[List[str], np.ndarray]) -> np.ndarray:
    """Standardizes input to a 2D byte array (|S1)."""
    if isinstance(data, list):
        if not data:
            return np.zeros((0, 0), dtype="|S1")
        return np.array([list(s) for s in data], dtype="|S1")

    # Handle Numpy Array inputs
    if data.size == 0:
        return np.zeros((0, 0), dtype="|S1")

    if data.ndim == 1:
        # For 1D array of strings, convert to 2D char array
        return np.array([list(s) for s in data], dtype="|S1")

    return data.astype("|S1")


def _get_blosum_params():
    """Lazy-initialization for BLOSUM parameters with neutralized Gap-vs-Gap."""
    if "char_to_int" not in _BLOSUM_CACHE:
        matrix_obj = substitution_matrices.load("BLOSUM62")
        alphabet = str(getattr(matrix_obj, "alphabet", "ARNDCQEGHILKMFPSTWYVBZX*"))
        lookup_table = np.array(matrix_obj)

        gap_idx = len(alphabet)

        # Default gap penalty for Gap-vs-Residue is -4.0
        final_lookup = np.full((gap_idx + 1, gap_idx + 1), -4.0, dtype=np.float64)
        final_lookup[:gap_idx, :gap_idx] = lookup_table

        # CRITICAL: Shared gaps are neutral (0.0)
        final_lookup[gap_idx, gap_idx] = BLOSUM_GAP_TO_GAP

        char_to_int = np.full(256, gap_idx, dtype=np.int32)
        for i, char in enumerate(alphabet):
            char_to_int[ord(char)] = i

        _BLOSUM_CACHE["char_to_int"] = char_to_int
        _BLOSUM_CACHE["final_lookup"] = final_lookup

    return _BLOSUM_CACHE["char_to_int"], _BLOSUM_CACHE["final_lookup"]
