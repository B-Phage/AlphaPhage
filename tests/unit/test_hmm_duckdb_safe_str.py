import numpy as np
import pandas as pd
import pytest
from hmm.duckdb_io import _duckdb_safe_str

# ----------------------------
# Missing / NA handling
# ----------------------------


@pytest.mark.parametrize(
    "value",
    [
        None,
        np.nan,
        pd.NA,
    ],
)
def test_duckdb_safe_str_missing_values(value):
    assert _duckdb_safe_str(value) is None


# ----------------------------
# Bytes-like inputs
# ----------------------------


@pytest.mark.parametrize(
    "value, expected",
    [
        (b"PF00001", "PF00001"),
        (bytearray(b"PF00002"), "PF00002"),
        (memoryview(b"PF00003"), "PF00003"),
    ],
)
def test_duckdb_safe_str_bytes_like(value, expected):
    assert _duckdb_safe_str(value) == expected


# ----------------------------
# Strings
# ----------------------------


def test_duckdb_safe_str_string_passthrough():
    s = "PF00004"
    out = _duckdb_safe_str(s)

    assert out == s
    assert out is s  # exact object preserved


# ----------------------------
# Numeric passthrough
# ----------------------------


@pytest.mark.parametrize(
    "value",
    [
        0,
        1,
        -5,
        3.14,
        np.int64(7),
        np.float64(2.71),
    ],
)
def test_duckdb_safe_str_numeric_passthrough(value):
    out = _duckdb_safe_str(value)

    assert out == value
    assert type(out) is type(value)


# ----------------------------
# Other objects are untouched
# ----------------------------


def test_duckdb_safe_str_other_object_passthrough():
    obj = {"a": 1}
    out = _duckdb_safe_str(obj)

    assert out is obj


# ----------------------------
# Pandas object-dtype safety check
# ----------------------------


def test_duckdb_safe_str_safe_for_dataframe_map():
    df = pd.DataFrame(
        {
            "a": [b"x", "y", None],
            "b": [1, 2, 3],
        }
    )

    for col in df.columns:
        df[col] = df[col].map(_duckdb_safe_str)

    assert df["a"].tolist() == ["x", "y", None]
    assert df["b"].tolist() == [1, 2, 3]
