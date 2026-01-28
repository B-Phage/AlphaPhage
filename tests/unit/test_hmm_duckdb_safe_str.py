import math

import pandas as pd
import pytest
from hmm.duckdb_io import assert_duckdb_safe, duckdb_force_string, duckdb_safe_numeric

# ---------------------------------------------------------------------
# duckdb_force_string
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "input_val, expected",
    [
        (None, None),
        (float("nan"), None),
        (b"bytes", "bytes"),
        (bytearray(b"bytearray"), "bytearray"),
        (memoryview(b"view"), "view"),
        (123, "123"),
        (3.14, "3.14"),
        ("already_str", "already_str"),
    ],
)
def test_duckdb_force_string(input_val, expected):
    assert duckdb_force_string(input_val) == expected


# ---------------------------------------------------------------------
# duckdb_safe_numeric
# ---------------------------------------------------------------------


def test_duckdb_safe_numeric_basic():
    series = pd.Series([1, 2.5, None, float("nan")])
    result = duckdb_safe_numeric(series)
    expected = pd.Series([1, 2.5, None, None])
    pd.testing.assert_series_equal(result, expected)


def test_duckdb_safe_numeric_all_invalid():
    series = pd.Series(["a", "b", None])
    result = duckdb_safe_numeric(series)
    expected = pd.Series([float("nan"), float("nan"), float("nan")])
    pd.testing.assert_series_equal(result, expected)


# ---------------------------------------------------------------------
# assert_duckdb_safe
# ---------------------------------------------------------------------


def test_assert_duckdb_safe_pass():
    df = pd.DataFrame({"str_col": ["a", "b", None], "num_col": [1.0, 2.5, None]})
    assert_duckdb_safe(df, numeric_cols=["num_col"])  # should not raise


def test_assert_duckdb_safe_fail_str():
    df = pd.DataFrame({"str_col": ["a", 1, None], "num_col": [1.0, 2.0, None]})
    with pytest.raises(TypeError, match="Column str_col contains non-str values"):
        assert_duckdb_safe(df, numeric_cols=["num_col"])


def test_assert_duckdb_safe_fail_numeric():
    df = pd.DataFrame(
        {"str_col": ["a", "b", None], "num_col": [1.0, "invalid", None]}  # mix invalid string
    )
    with pytest.raises(TypeError, match="Column num_col contains non-numeric values"):
        assert_duckdb_safe(df, numeric_cols=["num_col"])
