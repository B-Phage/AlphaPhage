from typing import Union, cast

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
        return _distance_matrix_identity_vectorized(alignment)
    elif matrix == "blosum62":
        return _distance_matrix_blosum62_vectorized(alignment)
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


def _distance_matrix_identity_vectorized(alignment: list[str] | np.ndarray) -> np.ndarray:
    """
    Calculates identity distance matrix using Numpy broadcasting.

    Args:
        alignment: A 2D numpy array of characters (shape: num_seqs x aln_length)

    Returns:
        2D numpy array of distances.
    """
    # TODO remove this after cc test
    # alignment_array = np.array([list(s) for s in alignment], dtype='U1')
    # np.save("challenging_alignment.npy", alignment_array)
    if len(alignment) == 0:
        return np.zeros((0, 0), dtype=np.float64)

    # Convert input to a 2D numpy character array if it isn't one
    # This ensures broadcasting works
    if isinstance(alignment, list):
        # Using dtype='U1' (Unicode 1-char) is memory efficient for alignments
        arr = np.array([list(s) for s in alignment], dtype="U1")
    else:
        arr = np.asarray(alignment)

    # Broadcasting for pairwise comparison
    # Resulting shape: (num_seqs, num_seqs, aln_len)
    matches = arr[:, None, :] == arr[None, :, :]

    # Calculate mean across the alignment length axis
    identity = np.mean(matches, axis=2, dtype=np.float64)

    # Explicitly cast back to ndarray to satisfy Pylance/Typing
    # np.atleast_2d ensures that even a 1x1 result is an array, not a scalar float
    result = np.atleast_2d(1.0 - identity)

    return result.astype(np.float64)


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
    """Distance based on BLOSUM62, normalized to [0,1] using Self-Score logic."""
    n = len(alignment)
    mat = np.zeros((n, n))
    blosum62 = cast(dict, substitution_matrices.load("BLOSUM62"))
    GAP_PENALTY = -4

    # Pre-calculate the "Self-Score" for every sequence
    # This is the score of a sequence compared against itself
    self_scores = []
    for seq in alignment:
        s_score = 0
        for char in seq:
            if char == "-":
                s_score += GAP_PENALTY
            else:
                # Get the score of the amino acid matching itself
                s_score += blosum62.get((char, char), blosum62.get((char, char), 0))
        self_scores.append(s_score)

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

            # --- THE CHANGE IS HERE ---
            # Compare the pair score to the best possible score of the two sequences
            max_possible = max(self_scores[i], self_scores[j])

            # If the max possible is 0 or less (all gaps), set distance to 1
            if max_possible <= 0:
                distance = 1.0
            else:
                distance = 1 - (score / max_possible)

            mat[i, j] = mat[j, i] = max(0.0, min(1.0, distance))

    return mat


def _distance_matrix_blosum62_vectorized(alignment: Union[list[str], np.ndarray]) -> np.ndarray:
    """Vectorized BLOSUM62 distance calculation.

    Assumes alignment is standardized: residues are uppercase,
    stops/gaps/dots are all '-' (gaps).
    """
    if len(alignment) == 0:
        return np.zeros((0, 0), dtype=np.float64)

    # 1. Convert to 2D numpy array of characters (if not already)
    arr = np.asarray([list(s) for s in alignment] if isinstance(alignment, list) else alignment)
    n_seqs, aln_len = arr.shape

    # 2. Prepare the substitution matrix and alphabet
    matrix_obj = substitution_matrices.load("BLOSUM62")
    # Biopython's BLOSUM62 alphabet usually includes 'X' and '*'
    alphabet = str(getattr(matrix_obj, "alphabet", "ARNDCQEGHILKMFPSTWYVBZX*"))
    lookup_table = np.array(matrix_obj)

    # 3. Handle the Gap '-'
    # Standard BLOSUM doesn't usually include '-', so we add a row/col for it.
    gap_idx = len(alphabet)
    final_lookup = np.full((gap_idx + 1, gap_idx + 1), -4.0, dtype=np.float64)
    final_lookup[:gap_idx, :gap_idx] = lookup_table

    # Optional: If two gaps align, some people prefer a score of 0 or +1.
    # Here we stick to -4 to match your previous iterative implementation.
    # final_lookup[gap_idx, gap_idx] = 1.0

    # 4. Fast Integer Mapping
    char_to_int = np.full(256, gap_idx)  # Default everything unknown to Gap index
    for i, char in enumerate(alphabet):
        char_to_int[ord(char)] = i

    # Convert characters to integers via ASCII view
    int_aln = char_to_int[arr.view("uint32")]

    # 5. Calculate Scores (Looping over positions to save memory)
    total_scores = np.zeros((n_seqs, n_seqs), dtype=np.float64)
    for pos in range(aln_len):
        col = int_aln[:, pos]
        total_scores += final_lookup[col[:, None], col[None, :]]

    # 6. Normalization (Self-Score similarity to distance)
    # Distance = 1 - (Score_ij / max(Score_ii, Score_jj))
    self_scores = np.diag(total_scores)
    max_self_scores = np.maximum.outer(self_scores, self_scores)

    # Safety against division by zero (e.g., all-gap sequences)
    max_self_scores[max_self_scores <= 0] = 1.0

    distance_matrix = 1.0 - (total_scores / max_self_scores)

    # Force range and symmetry
    distance_matrix = np.clip(distance_matrix, 0, 1)
    return (distance_matrix + distance_matrix.T) / 2
