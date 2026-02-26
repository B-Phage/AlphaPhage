import pandas as pd
import pytest
from variants.preprocessing_methods import (
    VALID_AA_CHARS,
    VALID_AMBIGUOUS_AA,
    _get_informative_columns,
    _group_consecutive_columns,
    split_names_diversity,
    standardize_frame_chars,
)

# ========================
# Fixtures
# ========================


@pytest.fixture
def sample_df():
    """Dataframe with lowercase, uppercase, '.', '*' and '-'"""
    return pd.DataFrame(
        {
            "col1": ["a", "A", "*", "."],
            "col2": ["c", "C", "C", "C"],
            "col3": ["d", "D", "D", "D"],
            "col4": ["-", "-", "-", "-"],
            "col5": ["*", "*", "*", "*"],
        }
    )


@pytest.fixture
def df_no_stops():
    """Dataframe without any stop codons"""
    return pd.DataFrame(
        {
            "col1": ["a", "A", "."],
            "col2": ["c", "C", "C"],
            "col3": ["d", "D", "D"],
            "col4": ["-", "-", "-"],
        }
    )


# ========================
# Fixtures for diversity/grouping tests
# ========================


@pytest.fixture
def simple_df():
    return pd.DataFrame(
        {
            "A": ["A", "A", "A"],
            "B": ["C", "C", "D"],  # informative
            "C": ["G", "G", "G"],
            "D": ["T", "T", "C"],  # informative
            "E": ["A", "A", "A"],
        }
    )


@pytest.fixture
def consecutive_informative_df():
    return pd.DataFrame(
        {
            "col1": ["A", "B", "C"],  # informative
            "col2": ["D", "E", "F"],  # informative
            "col3": ["G", "G", "G"],
            "col4": ["H", "I", "H"],  # informative
            "col5": ["J", "J", "J"],  # not informative
            "col6": ["K", "L", "M"],  # informative
            "col7": ["N", "N", "N"],
        }
    )


# ========================
# Tests
# ========================


def test_standardize_basic_replacement(sample_df):
    """Test uppercase conversion, '.' -> '-', '*' replaced with filler, stop metadata"""
    filler = "-"
    df_clean, stop_list = standardize_frame_chars(sample_df, filler_char=filler)

    # All stop codon rows tracked
    assert stop_list == [1, 1, 1, 1]

    # All entries are uppercase strings
    for val in df_clean.values.flatten():
        assert isinstance(val, str)
        # Only assert uppercase for alphabetic characters
        if val.isalpha():
            assert val.isupper()

    # '.' replaced by '-'
    assert df_clean.loc[3, "col1"] == "-"

    # '*' replaced by filler
    assert df_clean.loc[0, "col5"] == filler


def test_standardize_replace_stop_false(sample_df):
    """Test that '*' remains if replace_stop_codons=False"""
    df_clean, stop_list = standardize_frame_chars(sample_df, replace_stop_codons=False)

    # Stop codon metadata still tracked
    assert stop_list == [1, 1, 1, 1]

    # '*' still present
    assert "*" in df_clean.values

    # '.' replaced by '-'
    assert df_clean.loc[3, "col1"] == "-"


def test_standardize_keep_stop_metadata_false(sample_df):
    """Test disabling stop metadata tracking"""
    df_clean, stop_list = standardize_frame_chars(sample_df, keep_stop_metadata=False)
    assert stop_list == [0, 0, 0, 0]


def test_standardize_no_stop_codons(df_no_stops):
    """Dataframe without any '*' characters"""
    filler = "-"
    df_clean, stop_list = standardize_frame_chars(df_no_stops, filler_char=filler)

    # Stop list all zeros
    assert stop_list == [0, 0, 0]

    # '.' replaced
    assert df_clean.loc[2, "col1"] == "-"


def test_standardize_invalid_chars_raises(sample_df):
    """Invalid character triggers ValueError"""
    df_invalid = sample_df.copy()
    df_invalid.loc[0, "col1"] = "Z"  # Not allowed

    with pytest.raises(ValueError, match="Invalid characters found"):
        standardize_frame_chars(df_invalid)


def test_standardize_filler_char_validation(df_no_stops):
    """Using filler_char that is a valid AA or ambiguous AA raises ValueError"""
    with pytest.raises(ValueError, match="filler_char 'A' cannot be a valid amino acid"):
        standardize_frame_chars(df_no_stops, filler_char="A")
    with pytest.raises(ValueError, match="filler_char 'X' cannot be a valid amino acid"):
        standardize_frame_chars(df_no_stops, filler_char="X")


def test_standardize_only_stops():
    """Dataframe with only '*' characters replaced by filler"""
    df_stops = pd.DataFrame({"col1": ["*", "*"], "col2": ["*", "*"]})
    filler = "_"
    df_clean, stop_list = standardize_frame_chars(df_stops, filler_char=filler)

    # All stop metadata tracked
    assert stop_list == [1, 1]

    # All '*' replaced
    for val in df_clean.values.flatten():
        assert val == filler


def test_standardize_empty_df():
    """Empty DataFrame returns empty cleaned df and empty stop list"""
    df_empty = pd.DataFrame()
    df_clean, stop_list = standardize_frame_chars(df_empty)
    assert df_clean.empty
    assert stop_list == []


def test_standardize_raises_typeerror_on_non_dataframe():
    bad_input = [["A", "C"], ["D", "E"]]  # list of lists

    with pytest.raises(TypeError) as excinfo:
        standardize_frame_chars(bad_input)  # type: ignore

    assert "Expected pd.DataFrame" in str(excinfo.value)


# ========================
# Tests for _get_informative_columns
# ========================


def test_get_informative_columns(simple_df):
    mask = _get_informative_columns(simple_df)
    # Only columns B and D are informative
    assert mask == [False, True, False, True, False]


def test_get_informative_columns_all_same():
    df = pd.DataFrame({"A": [1, 1, 1], "B": ["X", "X", "X"]})
    mask = _get_informative_columns(df)
    assert mask == [False, False]


def test_get_informative_columns_all_unique():
    df = pd.DataFrame({"A": [1, 2, 3], "B": ["X", "Y", "Z"]})
    mask = _get_informative_columns(df)
    assert mask == [True, True]


# ========================
# Tests for _group_consecutive_columns
# ========================


def test_group_consecutive_columns_basic():
    columns = ["A", "B", "C", "D", "E"]
    mask = [False, True, False, True, False]
    groups = _group_consecutive_columns(columns, mask)
    assert groups == [["B"], ["D"]]


def test_group_consecutive_columns_consecutive():
    columns = ["A", "B", "C", "D"]
    mask = [True, True, False, True]
    groups = _group_consecutive_columns(columns, mask)
    assert groups == [["A", "B"], ["D"]]


def test_group_consecutive_columns_all_false():
    columns = ["A", "B", "C"]
    mask = [False, False, False]
    groups = _group_consecutive_columns(columns, mask)
    assert groups == []


def test_group_consecutive_columns_all_true():
    columns = ["A", "B", "C"]
    mask = [True, True, True]
    groups = _group_consecutive_columns(columns, mask)
    assert groups == [["A", "B", "C"]]


def test_group_consecutive_columns_empty():
    groups = _group_consecutive_columns([], [])
    assert groups == []


# ========================
# Tests for split_names_diversity
# ========================


def test_split_names_diversity_simple(simple_df):
    groups = split_names_diversity(simple_df)
    # B and D are single informative columns
    assert groups == [["B"], ["D"]]


def test_split_names_diversity_consecutive(consecutive_informative_df):
    groups = split_names_diversity(consecutive_informative_df)
    # col1 & col2 are consecutive informative, col4 alone, col6 alone
    assert groups == [["col1", "col2"], ["col4"], ["col6"]]


def test_split_names_diversity_all_same():
    df = pd.DataFrame({"A": [1, 1, 1], "B": ["X", "X", "X"]})
    groups = split_names_diversity(df)
    # No informative columns
    assert groups == []


def test_split_names_diversity_all_informative():
    df = pd.DataFrame({"A": [1, 2, 3], "B": ["X", "Y", "Z"], "C": [4, 5, 6]})
    groups = split_names_diversity(df)
    # All columns are consecutive informative
    assert groups == [["A", "B", "C"]]
