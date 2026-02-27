# test_align_table.py

from pathlib import Path

import pandas as pd
import pytest
from omegaconf import DictConfig, OmegaConf
from representation.align_table import (
    _align_rows,
    process_msa_table,
    process_table,
    read_table,
    select_columns,
    split_index_column,
)

# -----------------------------
# Fixtures
# -----------------------------


@pytest.fixture
def vocab_config():
    # Minimal vocab with filler_char for tests
    cfg = DictConfig({"filler_char": "_"})
    return cfg


@pytest.fixture
def sample_csv(tmp_path):
    content = "id,A,B,C\nrow1,1,2,3\nrow2,4,5,6"
    path = tmp_path / "sample.csv"
    path.write_text(content)
    return path


@pytest.fixture
def sample_tsv(tmp_path):
    content = "id\tA\tB\tC\nrow1\t1\t2\t3\nrow2\t4\t5\t6"
    path = tmp_path / "sample.tsv"
    path.write_text(content)
    return path


@pytest.fixture
def sample_csa(tmp_path):
    # Simple alignment with sequences and indices
    content = "ACGT,row1;prot1\nTGCA,row2;prot2"
    path = tmp_path / "sample.csa"
    path.write_text(content)
    return path


@pytest.fixture
def row_names_list():
    return ["row1", "row2", "row3"]  # row3 missing


@pytest.fixture
def col_names_list():
    return ["A", "C"]


# -----------------------------
# Tests for read_table
# -----------------------------


def test_read_csv(sample_csv):
    df = read_table(str(sample_csv), "csv")
    assert list(df.index) == ["row1", "row2"]
    assert list(df.columns) == ["A", "B", "C"]
    assert df.loc["row1", "B"] == 2


def test_read_tsv(sample_tsv):
    df = read_table(str(sample_tsv), "tsv")
    assert list(df.index) == ["row1", "row2"]
    assert list(df.columns) == ["A", "B", "C"]
    assert df.loc["row2", "C"] == 6


def test_read_csa(sample_csa):
    df = read_table(str(sample_csa), "csa")
    # Index should be the second column
    assert list(df.index) == ["row1;prot1", "row2;prot2"]
    # Columns should match sequence positions
    assert df.iloc[0, 0] == "A"
    assert df.iloc[1, 3] == "A"


def test_read_table_invalid_format(sample_csv):
    import pytest

    with pytest.raises(ValueError):
        read_table(str(sample_csv), "invalid")


# -----------------------------
# Tests for split_index_column
# -----------------------------


def test_split_index_column_basic():
    df = pd.DataFrame({"val": [1, 2]}, index=["rowA;prot1", "rowB;prot2"])
    df2 = split_index_column(df, ";", 0)
    assert list(df2.index) == ["rowA", "rowB"]


def test_split_index_column_second_element():
    df = pd.DataFrame({"val": [1, 2]}, index=["rowA;prot1", "rowB;prot2"])
    df2 = split_index_column(df, ";", 1)
    assert list(df2.index) == ["prot1", "prot2"]


# -----------------------------
# Tests for _align_rows
# -----------------------------


def test_align_rows_adds_missing(vocab_config):
    df = pd.DataFrame({"A": [1], "B": [2]}, index=["row1"])
    df2 = _align_rows(df, ["row1", "row2"], vocab_config.filler_char)
    assert "row1" in df2.index
    assert "row2" in df2.index
    # missing row filled with filler_char
    assert all(df2.loc["row2"] == "_")


# -----------------------------
# Tests for select_columns
# -----------------------------


def test_select_columns_keep_all():
    df = pd.DataFrame({"A": [1], "B": [2]})
    df2 = select_columns(df, None)
    assert list(df2.columns) == ["A", "B"]


def test_select_columns_subset():
    df = pd.DataFrame({"A": [1], "B": [2], "C": [3]})
    df2 = select_columns(df, ["A", "C"])
    assert list(df2.columns) == ["A", "C"]


def test_select_columns_none_in_list_raises():
    df = pd.DataFrame({"A": [1], "B": [2]})
    import pytest

    with pytest.raises(ValueError):
        select_columns(df, [None, "A"])


def test_select_columns_newline_raises():
    df = pd.DataFrame({"A": [1], "B": [2]})
    import pytest

    with pytest.raises(ValueError):
        select_columns(df, ["A\n"])


# -----------------------------
# Tests for process_table
# -----------------------------


def test_process_table_csv(sample_csv, vocab_config, row_names_list, col_names_list):
    df = process_table(str(sample_csv), "csv", vocab_config, row_names_list, col_names_list)
    # row3 added as missing
    assert "row3" in df.index
    assert all(df.loc["row3"] == vocab_config.filler_char)
    # only selected columns
    assert list(df.columns) == col_names_list


def test_process_table_splits_index(tmp_path):
    """
    Ensure that index splitting occurs when index_element is not None.
    """
    # Create mock CSV file
    data = pd.DataFrame(
        {"col1": [1, 2], "col2": [3, 4]},
        index=["genomeA;protein1", "genomeB;protein2"],
    )
    file_path = tmp_path / "test.csv"
    data.to_csv(file_path)

    vocab = OmegaConf.create({"filler_char": "-"})

    df = process_table(
        in_file=str(file_path),
        file_format="csv",
        vocab=vocab,
        index_element=0,  # <-- branch under test
        sep_index_split=";",
    )

    assert list(df.index) == ["genomeA", "genomeB"]


def test_process_table_splits_index_second_element(tmp_path):
    """
    Ensure correct element of split index is retained.
    """
    data = pd.DataFrame(
        {"col1": [10, 20]},
        index=["genomeA;protein1", "genomeB;protein2"],
    )
    file_path = tmp_path / "test2.csv"
    data.to_csv(file_path)

    vocab = OmegaConf.create({"filler_char": "-"})

    df = process_table(
        in_file=str(file_path),
        file_format="csv",
        vocab=vocab,
        index_element=1,
        sep_index_split=";",
    )

    assert list(df.index) == ["protein1", "protein2"]


def test_process_table_split_then_align(tmp_path):
    """
    Ensure index is split before row alignment.
    Missing rows should be filled using filler_char.
    """
    data = pd.DataFrame(
        {"col1": [1]},
        index=["genomeA;protein1"],
    )
    file_path = tmp_path / "test3.csv"
    data.to_csv(file_path)

    vocab = OmegaConf.create({"filler_char": "X"})

    df = process_table(
        in_file=str(file_path),
        file_format="csv",
        vocab=vocab,
        index_element=0,
        sep_index_split=";",
        row_names=["genomeA", "genomeB"],  # genomeB missing
    )

    # Row order must match row_names
    assert list(df.index) == ["genomeA", "genomeB"]

    # genomeB should be filler
    assert (df.loc["genomeB"] == "X").all()


def test_process_table_split_index_out_of_range(tmp_path):
    """
    Ensure IndexError is raised if index_element is invalid.
    """
    data = pd.DataFrame(
        {"col1": [1]},
        index=["genomeA;protein1"],
    )
    file_path = tmp_path / "test4.csv"
    data.to_csv(file_path)

    vocab = OmegaConf.create({"filler_char": "-"})

    with pytest.raises(IndexError):
        process_table(
            in_file=str(file_path),
            file_format="csv",
            vocab=vocab,
            index_element=2,  # invalid
            sep_index_split=";",
        )


# -----------------------------
# Tests for process_msa_table
# -----------------------------


def test_process_msa_table(sample_csa, vocab_config, row_names_list):
    df = process_msa_table(sample_csa, vocab_config, row_names_list)
    # row3 added as missing
    assert "row3" in df.index
    # All characters are str
    assert df.apply(lambda col: col.map(lambda x: isinstance(x, str))).all().all()


def test_process_msa_table_raises_for_wrong_extension(tmp_path):
    """
    Ensure ValueError is raised when file extension is not .csa.
    """
    file_path = tmp_path / "test.csv"
    file_path.write_text("dummy,data\n")

    vocab = OmegaConf.create({"filler_char": "-"})

    with pytest.raises(ValueError, match="Expected a .csa MSA file"):
        process_msa_table(
            msa_file=file_path,
            vocab=vocab,
        )
