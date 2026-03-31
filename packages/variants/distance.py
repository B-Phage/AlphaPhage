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
    if q_arr.size == 0 or t_arr.size == 0:
        return np.zeros((q_arr.shape[0], t_arr.shape[0]), dtype=np.float64)

    # 1. Cast to uint8 immediately to ensure integer comparisons work (45, 46)
    q_val = q_arr.view(np.uint8)
    t_val = t_arr.view(np.uint8)

    # 2. Expand dimensions (M, N, L)
    q_exp = q_val[:, None, :]
    t_exp = t_val[None, :, :]

    # 3. Identify gaps (45 is '-', 46 is '.')
    # By using uint8, we guarantee these integer comparisons are accurate
    is_gap_q = (q_exp == 45) | (q_exp == 46)
    is_gap_t = (t_exp == 45) | (t_exp == 46)

    # 4. Define Shared Gaps
    shared_mask = is_gap_q & is_gap_t

    # 5. Matches: MUST be identical AND MUST NOT be a shared gap
    # This prevents '-' == '-' from being counted as a match
    match_mask = (q_exp == t_exp) & ~shared_mask
    real_match_counts = np.sum(match_mask, axis=2)

    # 6. Comparable positions: Total length - shared gaps
    total_comparable = q_val.shape[1] - np.sum(shared_mask, axis=2)
    total_comparable = np.maximum(total_comparable, 1)

    identity = real_match_counts / total_comparable
    return np.clip(1.0 - identity, 0, 1)


def _engine_blosum62(
    q_arr: np.ndarray, t_arr: np.ndarray, c2i: np.ndarray, lookup: np.ndarray
) -> np.ndarray:
    """Calculates BLOSUM62 Distance with Geometric Mean Normalization."""
    if q_arr.size == 0 or t_arr.size == 0:
        return np.zeros((q_arr.shape[0], t_arr.shape[0]), dtype=np.float64)

    q_int = c2i[q_arr.view("uint8")]
    t_int = c2i[t_arr.view("uint8")]

    gap_idx = lookup.shape[0] - 1
    M, L = q_int.shape
    N = t_int.shape[0]

    # 1. Vectorized Score Calculation
    total_scores = np.zeros((M, N), dtype=np.float64)
    for l in range(L):
        total_scores += lookup[q_int[:, l, None], t_int[None, :, l]]

    # 2. Correct Shared Gap Logic for Normalization
    is_gap_q = q_int == gap_idx
    is_gap_t = t_int == gap_idx

    # Use bitwise AND in 3D to find gaps at the same index
    shared_gap_counts = np.sum(is_gap_q[:, None, :] & is_gap_t[None, :, :], axis=2)

    # 3. Calculate Normalized Self-Scores
    diag = np.diag(lookup)
    gap_self_score = diag[gap_idx]

    q_self_total = np.sum(diag[q_int], axis=1, keepdims=True)
    t_self_total = np.sum(diag[t_int], axis=1, keepdims=True).T

    # Subtract the self-score contributed by shared gaps
    q_self_masked = q_self_total - (shared_gap_counts * gap_self_score)
    t_self_masked = t_self_total - (shared_gap_counts * gap_self_score)

    norm_factor = np.sqrt(np.maximum(q_self_masked * t_self_masked, 1.0))
    distance = 1.0 - (total_scores / norm_factor)

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
    """Lazy-initialization with relaxed linear gap penalty."""
    if "char_to_int" not in _BLOSUM_CACHE:
        matrix_obj = substitution_matrices.load("BLOSUM62")
        alphabet = str(getattr(matrix_obj, "alphabet", "ARNDCQEGHILKMFPSTWYVBZX*"))
        lookup_table = np.array(matrix_obj)

        gap_idx = len(alphabet)

        # REFACTORED: Changed from -4.0 to -2.0 to mimic affine extension forgiveness
        RELAXED_GAP_PENALTY = -2.0

        final_lookup = np.full((gap_idx + 1, gap_idx + 1), RELAXED_GAP_PENALTY, dtype=np.float64)
        final_lookup[:gap_idx, :gap_idx] = lookup_table

        # Shared gaps remain neutral
        final_lookup[gap_idx, gap_idx] = 0.0

        char_to_int = np.full(256, gap_idx, dtype=np.int32)
        for i, char in enumerate(alphabet):
            char_to_int[ord(char.upper())] = i
            char_to_int[ord(char.lower())] = i  # Robustness fix

        _BLOSUM_CACHE["char_to_int"] = char_to_int
        _BLOSUM_CACHE["final_lookup"] = final_lookup

    return _BLOSUM_CACHE["char_to_int"], _BLOSUM_CACHE["final_lookup"]
