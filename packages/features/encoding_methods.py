# features/encoding_methods.py

from typing import Dict, Iterable, List, Optional


def encode_onecol_vocab_vec(
    vec: Iterable[str], vocab_dict: Dict[str, int], unk_token: Optional[str] = None
) -> List[List[int]]:
    """
    Encode a single column/vector using a vocabulary mapping.

    Each element in `vec` is replaced by its corresponding integer in `vocab_dict`.
    Returns a nested list [[...]] to be compatible with downstream clustering/feature pipelines.

    Parameters
    ----------
    vec : Iterable[str]
        Iterable of symbols to encode (e.g., cluster labels, amino acids, tokens).
    vocab_dict : Dict[str, int]
        Dictionary mapping symbols to integer indices.
    unk_token : str, optional
        Token to use if an element in vec is not in vocab_dict.
        If None, missing elements will raise a KeyError.

    Returns
    -------
    List[List[int]]
        Nested list of encoded integers with shape [1, len(vec)].
    """
    encoded = []
    for e in vec:
        if e in vocab_dict:
            encoded.append(vocab_dict[e])
        elif unk_token is not None and unk_token in vocab_dict:
            encoded.append(vocab_dict[unk_token])
        else:
            raise KeyError(f"Token '{e}' not found in vocab_dict and no unk_token provided.")
    return [encoded]  # wrap in a list to match [[...]] format used in cluster features


# Optional: alias for clarity in pipelines
encode_column = encode_onecol_vocab_vec
