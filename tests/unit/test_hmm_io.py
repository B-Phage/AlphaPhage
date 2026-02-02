from unittest.mock import MagicMock, patch

# Adjust this import to match your project structure
import hmm.io as hmm_io
import pandas as pd
import pytest

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


class DummyHMM:
    def __init__(self, name=None, accession=None):
        self.name = name
        self.accession = accession
        self.write = MagicMock()


class DummyHMMFile:
    """
    Context-manager compatible fake HMMFile.
    """

    def __init__(self, hmms=None, pressed=False):
        self._hmms = hmms or []
        self._pressed = pressed

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        pass

    def read(self):
        return self._hmms[0] if self._hmms else None

    def __iter__(self):
        return iter(self._hmms)

    def is_pressed(self):
        return self._pressed

    def optimized_profiles(self):
        return iter(self._hmms)


# ---------------------------------------------------------------------
# load_single_hmm_per_file
# ---------------------------------------------------------------------


def test_load_single_hmm_per_file_success():
    dummy_hmm = DummyHMM(name=b"test_hmm")

    with (
        patch("hmm.io.glob.glob", return_value=["/tmp/a.hmm", "/tmp/b.hmm"]),
        patch("pyhmmer.plan7.HMMFile", return_value=DummyHMMFile([dummy_hmm])),
    ):

        hmms = hmm_io.load_single_hmm_per_file("/tmp", "*.hmm")

    assert len(hmms) == 2
    assert all(hmm is dummy_hmm for hmm in hmms)


def test_load_single_hmm_per_file_no_files():
    with patch("hmm.io.glob.glob", return_value=[]), patch("hmm.io.logger") as mock_logger:

        with pytest.raises(ValueError, match="No HMMs found"):
            hmm_io.load_single_hmm_per_file("/tmp", "*.hmm")

        mock_logger.error.assert_called_once()


# ---------------------------------------------------------------------
# load_hmms_from_file
# ---------------------------------------------------------------------


def test_load_hmms_from_file_success():
    hmms = [DummyHMM(name=b"hmm1"), DummyHMM(name=b"hmm2")]

    with patch("pyhmmer.plan7.HMMFile", return_value=DummyHMMFile(hmms)):
        result = hmm_io.load_hmms_from_file("/tmp/multi.hmm")

    assert result == hmms


# ---------------------------------------------------------------------
# split_hmms
# ---------------------------------------------------------------------


def test_split_hmms_writes_named_files(tmp_path):
    hmms = [
        DummyHMM(name=b"hmmA"),
        DummyHMM(name=b"hmmB"),
    ]

    with patch("hmm.io.load_hmms_from_file", return_value=hmms):
        hmm_io.split_hmms("/tmp/input.hmm", tmp_path)

    for hmm in hmms:
        assert hmm.name is not None
        filename = tmp_path / f"{hmm.name.decode()}.hmm"
        assert filename.exists()
        hmm.write.assert_called_once()


def test_split_hmms_uses_accession_if_name_missing(tmp_path):
    hmm = DummyHMM(name=None, accession=b"ACC123")

    with patch("hmm.io.load_hmms_from_file", return_value=[hmm]):
        hmm_io.split_hmms("/tmp/input.hmm", tmp_path)

    expected_file = tmp_path / "ACC123.hmm"
    assert expected_file.exists()
    hmm.write.assert_called_once()


def test_split_hmms_fallback_name_and_warning(tmp_path):
    hmm = DummyHMM(name=None, accession=None)

    with (
        patch("hmm.io.load_hmms_from_file", return_value=[hmm]),
        patch("hmm.io.logger") as mock_logger,
    ):

        hmm_io.split_hmms("/tmp/input.hmm", tmp_path)

        mock_logger.warning.assert_called_once()
        assert (tmp_path / "unnamed_hmm_0.hmm").exists()


def test_split_hmms_write_error_logged(tmp_path):
    hmm = DummyHMM(name=b"bad_hmm")
    hmm.write.side_effect = IOError("disk error")

    with (
        patch("hmm.io.load_hmms_from_file", return_value=[hmm]),
        patch("hmm.io.logger") as mock_logger,
    ):

        hmm_io.split_hmms("/tmp/input.hmm", tmp_path)

        mock_logger.error.assert_called_once()


# ---------------------------------------------------------------------
# load_pressed_hmms
# ---------------------------------------------------------------------


def test_load_pressed_hmms_success():
    hmms = [DummyHMM(name=b"hmm1"), DummyHMM(name=b"hmm2")]

    with patch(
        "pyhmmer.plan7.HMMFile",
        return_value=DummyHMMFile(hmms, pressed=True),
    ):
        result = hmm_io.load_pressed_hmms("/tmp/pressed.hmm")
    assert result is not None
    assert list(result) == hmms


def test_load_pressed_hmms_not_pressed():
    with (
        patch(
            "pyhmmer.plan7.HMMFile",
            return_value=DummyHMMFile([], pressed=False),
        ),
        patch("hmm.io.logger") as mock_logger,
    ):

        with pytest.raises(ValueError):
            hmm_io.load_pressed_hmms("/tmp/not_pressed.hmm")

        mock_logger.error.assert_called_once()


def test_write_hmm_scan_summary(tmp_path):
    # 1. Setup Mock Data
    data = {
        "gene": ["A", "B", "C", "A"],  # 3 unique
        "score": [10, 20, 30, 40],  # 4 unique
        "large_col": [f"val_{i}" for i in range(25)],  # 25 unique
    }
    # Adjusting data length to match the longest column
    df = pd.DataFrame(
        {
            "gene": ["A", "B", "C", "A"] + [None] * 21,
            "score": list(range(25)),
            "large_col": [f"val_{i}" for i in range(25)],
        }
    )

    out_file = tmp_path / "summary.csv"
    unique_cols = ("gene", "large_col")

    # 2. Execute
    hmm_io.write_hmm_scan_summary(
        df, out_csv=out_file, unique_cols=unique_cols, max_unique_preview=10
    )

    # 3. Assertions
    assert out_file.exists()
    result_df = pd.read_csv(out_file)

    # Check shape metrics
    assert result_df.loc[0, "n_rows"] == 25
    assert result_df.loc[0, "n_columns"] == 3

    # Check gene column (Under threshold)
    # Note: nunique(dropna=True) should be 3 (A, B, C)
    assert result_df.loc[0, "gene_nunique"] == 3
    # Check preview contains the expected strings (order might vary, so we check set)
    raw_preview = str(result_df.loc[0, "gene_unique_preview"])
    preview_values = set(raw_preview.split(","))
    assert preview_values == {"A", "B", "C"}

    # Check large_col (Over threshold)
    assert result_df.loc[0, "large_col_nunique"] == 25
    assert result_df.loc[0, "large_col_unique_preview"] == ">10"


def test_write_hmm_scan_summary_empty_cols(tmp_path):
    """Test behavior when no unique_cols are provided."""
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    out_file = tmp_path / "empty_summary.csv"

    hmm_io.write_hmm_scan_summary(df, out_file)

    result_df = pd.read_csv(out_file)
    assert list(result_df.columns) == ["n_rows", "n_columns"]
    assert result_df.loc[0, "n_rows"] == 2


def test_split_hmms_str_name(tmp_path):
    hmms = [DummyHMM(name="hmmStr")]  # str instead of bytes
    with patch("hmm.io.load_hmms_from_file", return_value=hmms):
        # Should not raise after fix
        hmm_io.split_hmms("/tmp/fake.hmm", tmp_path)
        # Assert the file was created correctly
        assert (tmp_path / "hmmStr.hmm").exists()
