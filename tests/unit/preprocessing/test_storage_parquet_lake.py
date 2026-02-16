from pathlib import Path

import pandas as pd
import pytest
from storage_io.parquet_lake import enforce_schema, write_parquet_partition


# ------------------------
# Fixtures / helpers
# ------------------------
@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {"protein_name": ["p1", "p2", "p3"], "length": [100, 200, 300], "score": [1.0, 2.0, 3.0]}
    )


@pytest.fixture
def schema():
    return {"protein_name": "string", "length": "int", "score": "float", "extra_col": "string"}


# ------------------------
# enforce_schema tests
# ------------------------
def test_enforce_schema_adds_missing_columns(sample_df, schema):
    # extra_col is missing in df
    df_aligned = enforce_schema(sample_df, schema)
    assert set(df_aligned.columns) == set(schema.keys())
    # missing column filled with nulls
    assert df_aligned["extra_col"].isna().all()


def test_enforce_schema_drops_extra_columns(sample_df, schema):
    df = sample_df.copy()
    df["unexpected"] = [1, 2, 3]
    df_aligned = enforce_schema(df, schema)
    assert "unexpected" not in df_aligned.columns


def test_enforce_schema_casts_types(sample_df, schema):
    df = sample_df.copy()
    df_aligned = enforce_schema(df, schema)
    assert pd.api.types.is_string_dtype(df_aligned["protein_name"])
    assert pd.api.types.is_integer_dtype(df_aligned["length"])
    assert pd.api.types.is_float_dtype(df_aligned["score"])
    assert pd.api.types.is_string_dtype(df_aligned["extra_col"])


# ------------------------
# write_parquet_partition tests
# ------------------------
def test_write_parquet_partition_creates_file(tmp_path, sample_df):
    out_file = tmp_path / "proteins.parquet"
    write_parquet_partition(sample_df, out_file, key_cols=["protein_name"])
    assert out_file.exists()

    # Read back and check content
    df_read = pd.read_parquet(out_file)
    assert set(df_read.columns) == set(sample_df.columns)
    assert len(df_read) == len(sample_df)


def test_write_parquet_partition_empty_df(tmp_path):
    out_file = tmp_path / "empty.parquet"
    df_empty = pd.DataFrame(columns=["protein_name", "length"])
    # Should not raise, but also not create a file
    write_parquet_partition(df_empty, out_file, key_cols=["protein_name"])
    assert not out_file.exists()


def test_write_parquet_partition_detects_duplicates(tmp_path, sample_df):
    out_file = tmp_path / "dup.parquet"
    df_dup = pd.concat([sample_df, sample_df.iloc[:1]], ignore_index=True)  # duplicate first row
    with pytest.raises(ValueError, match="Duplicate keys detected"):
        write_parquet_partition(df_dup, out_file, key_cols=["protein_name"])


def test_write_parquet_partition_creates_parents(tmp_path, sample_df):
    out_file = tmp_path / "nested" / "dir" / "proteins.parquet"
    write_parquet_partition(sample_df, out_file, key_cols=["protein_name"])
    assert out_file.exists()
    # check parent dirs exist
    assert (tmp_path / "nested" / "dir").exists()
