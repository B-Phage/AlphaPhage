import pytest
from features.encoding_methods import (
    encode_column,
    encode_onecol_vocab_vec,
)

# ---------------------------------------------------------
# Basic Happy Path
# ---------------------------------------------------------


def test_basic_encoding():
    vec = ["A", "B", "C"]
    vocab = {"A": 0, "B": 1, "C": 2}

    result = encode_onecol_vocab_vec(vec, vocab)

    assert result == [[0, 1, 2]]
    assert isinstance(result, list)
    assert isinstance(result[0], list)


def test_order_is_preserved():
    vec = ["C", "A", "B"]
    vocab = {"A": 0, "B": 1, "C": 2}

    result = encode_onecol_vocab_vec(vec, vocab)

    assert result == [[2, 0, 1]]


# ---------------------------------------------------------
# Unknown Token Handling
# ---------------------------------------------------------


def test_unknown_token_with_unk_token_present():
    vec = ["A", "X", "B"]
    vocab = {"A": 0, "B": 1, "<UNK>": 99}

    result = encode_onecol_vocab_vec(vec, vocab, unk_token="<UNK>")

    assert result == [[0, 99, 1]]


def test_unknown_token_without_unk_token_raises():
    vec = ["A", "X"]
    vocab = {"A": 0}

    with pytest.raises(KeyError, match="Token 'X' not found"):
        encode_onecol_vocab_vec(vec, vocab)


def test_unk_token_provided_but_not_in_vocab_raises():
    vec = ["A", "X"]
    vocab = {"A": 0}  # no <UNK> in vocab

    with pytest.raises(KeyError):
        encode_onecol_vocab_vec(vec, vocab, unk_token="<UNK>")


# ---------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------


def test_empty_vector_returns_nested_empty():
    vec = []
    vocab = {"A": 0}

    result = encode_onecol_vocab_vec(vec, vocab)

    assert result == [[]]


def test_accepts_tuple_input():
    vec = ("A", "B")
    vocab = {"A": 0, "B": 1}

    result = encode_onecol_vocab_vec(vec, vocab)

    assert result == [[0, 1]]


def test_accepts_generator_input():
    vec = (x for x in ["A", "B"])
    vocab = {"A": 0, "B": 1}

    result = encode_onecol_vocab_vec(vec, vocab)

    assert result == [[0, 1]]


# ---------------------------------------------------------
# Alias Behaviour
# ---------------------------------------------------------


def test_encode_column_alias():
    vec = ["A"]
    vocab = {"A": 42}

    result = encode_column(vec, vocab)

    assert result == [[42]]


# ---------------------------------------------------------
# Input Immutability
# ---------------------------------------------------------


def test_vocab_not_modified():
    vec = ["A"]
    vocab = {"A": 0}

    _ = encode_onecol_vocab_vec(vec, vocab)

    assert vocab == {"A": 0}


def test_vec_not_modified():
    vec = ["A", "B"]
    original = vec.copy()
    vocab = {"A": 0, "B": 1}

    _ = encode_onecol_vocab_vec(vec, vocab)

    assert vec == original
