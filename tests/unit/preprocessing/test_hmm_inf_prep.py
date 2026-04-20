from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from hmm.inference_prep import (
    filter_dat_annotations,
    filter_phrog_annotations,
    get_selected_hmm_set,
    prepare_inference_hmm_set,
)

## --- Fixtures ---


@pytest.fixture
def temp_dir(tmp_path):
    """Provides a temporary directory for file operations."""
    return tmp_path


@pytest.fixture
def mock_model_tsv(temp_dir):
    """Creates a dummy TSV with HMM names in the first line."""
    tsv_path = temp_dir / "model_selection.tsv"
    tsv_path.write_text("phrog_1\tphrog_2\tpfam_A\nignore_this_row", encoding="utf-8")
    return tsv_path


## --- Tests for get_selected_hmm_set ---


def test_get_selected_hmm_set_success(mock_model_tsv):
    result = get_selected_hmm_set(mock_model_tsv)
    assert result == {"phrog_1", "phrog_2", "pfam_A"}


def test_get_selected_hmm_set_empty(temp_dir):
    empty_file = temp_dir / "empty.tsv"
    empty_file.write_text("", encoding="utf-8")
    assert get_selected_hmm_set(empty_file) == set()


def test_get_selected_hmm_set_not_found():
    with pytest.raises(FileNotFoundError):
        get_selected_hmm_set("non_existent_file.tsv")


## --- Tests for prepare_inference_hmm_set ---


@patch("pyhmmer.plan7.HMMFile")
@patch("pyhmmer.hmmer.hmmpress")
def test_prepare_inference_hmm_set_as_pressed(mock_press, mock_hmmfile, temp_dir, mock_model_tsv):
    # Setup: Create dummy .hmm files
    unpacked = temp_dir / "unpacked"
    unpacked.mkdir()
    (unpacked / "phrog_1.hmm").write_text("dummy content")
    (unpacked / "phrog_2.hmm").write_text("dummy content")

    # Mock the HMM object reading
    mock_hmm_obj = MagicMock()
    mock_hmmfile.return_value.__enter__.return_value.read.return_value = mock_hmm_obj

    out_path = temp_dir / "output.h3m"

    prepare_inference_hmm_set(unpacked, mock_model_tsv, out_path, as_pressed=True)

    # Verify hmmpress was called with the collected HMMs
    assert mock_press.called
    assert len(mock_press.call_args[0][0]) == 2  # phrog_1 and phrog_2


def test_prepare_inference_hmm_set_no_hmms_found(temp_dir, mock_model_tsv, caplog):
    # Empty directory
    unpacked = temp_dir / "empty_dir"
    unpacked.mkdir()
    out_path = temp_dir / "out.hmm"

    prepare_inference_hmm_set(unpacked, mock_model_tsv, out_path, as_pressed=False)
    assert "No HMMs were successfully loaded" in caplog.text


@patch("hmm.inference_prep.get_selected_hmm_set")
def test_prepare_inference_hmm_set_exception(mock_get_set, temp_dir):
    mock_get_set.side_effect = Exception("TSV Error")
    with pytest.raises(Exception, match="TSV Error"):
        prepare_inference_hmm_set(temp_dir, "fake.tsv", "out.hmm")


## --- Tests for filter_dat_annotations ---


def test_filter_dat_annotations_success(temp_dir, mock_model_tsv):
    input_dat = temp_dir / "input.dat"
    input_dat.write_text(
        "#=GF ID   phrog_1\nSome data\n//\n"
        "#=GF ID   skip_me\nOther data\n//\n"
        "#=GF ID   phrog_2\nMore data\n//\n",
        encoding="utf-8",
    )
    output_dat = temp_dir / "filtered.dat"

    count = filter_dat_annotations(input_dat, output_dat, mock_model_tsv)

    assert count == 2
    content = output_dat.read_text()
    assert "phrog_1" in content
    assert "phrog_2" in content
    assert "skip_me" not in content


## --- Tests for filter_phrog_annotations ---


def test_filter_phrog_annotations_success(temp_dir, mock_model_tsv):
    # mock_model_tsv contains phrog_1 and phrog_2
    input_csv = temp_dir / "ann.tsv"
    data = {"phrog": ["1", "2", "3"], "other": ["a", "b", "c"]}
    pd.DataFrame(data).to_csv(input_csv, sep="\t", index=False)

    output_csv = temp_dir / "filtered_ann.tsv"
    count = filter_phrog_annotations(input_csv, output_csv, mock_model_tsv)

    assert count == 2
    df = pd.read_csv(output_csv, sep="\t")
    assert set(df["phrog"].astype(str)) == {"1", "2"}


def test_filter_phrog_annotations_bad_format(temp_dir, caplog):
    # Create TSV with a name missing the underscore
    bad_tsv = temp_dir / "bad.tsv"
    bad_tsv.write_text("nounderscore", encoding="utf-8")

    input_csv = temp_dir / "ann.tsv"
    pd.DataFrame({"phrog": ["1"]}).to_csv(input_csv, sep="\t", index=False)

    filter_phrog_annotations(input_csv, temp_dir / "out.tsv", bad_tsv)
    assert "Unexpected HMM name format" in caplog.text


## --- Additional tests for 100% Coverage ---


@patch("pyhmmer.plan7.HMMFile")
def test_prepare_inference_hmm_set_monolithic_text(mock_hmmfile, temp_dir, mock_model_tsv):
    """Hits the 'else' (as_pressed=False) and the 'missing_count == 0' log."""
    unpacked = temp_dir / "unpacked"
    unpacked.mkdir()

    # Create all files requested in mock_model_tsv (phrog_1, phrog_2, pfam_A)
    for name in ["phrog_1", "phrog_2", "pfam_A"]:
        (unpacked / f"{name}.hmm").write_text("dummy")

    mock_hmm_obj = MagicMock()
    mock_hmmfile.return_value.__enter__.return_value.read.return_value = mock_hmm_obj

    out_path = temp_dir / "monolith.hmm"
    # as_pressed=False triggers the 'with open(...) as dst' block
    prepare_inference_hmm_set(unpacked, mock_model_tsv, out_path, as_pressed=False)

    assert out_path.exists()
    assert mock_hmm_obj.write.called


@patch("pyhmmer.plan7.HMMFile")
def test_prepare_inference_hmm_set_partial_missing(mock_hmmfile, temp_dir, mock_model_tsv, caplog):
    """Hits the 'if missing_count > 0' warning at the end."""
    unpacked = temp_dir / "unpacked"
    unpacked.mkdir()

    # Only create ONE of the requested HMMs
    (unpacked / "phrog_1.hmm").write_text("dummy")

    mock_hmmfile.return_value.__enter__.return_value.read.return_value = MagicMock()

    prepare_inference_hmm_set(unpacked, mock_model_tsv, temp_dir / "out.hmm", as_pressed=False)

    assert "packaged, 2 missing" in caplog.text


def test_filter_phrog_annotations_exception(temp_dir):
    """Hits the try-except block in filter_phrog_annotations."""
    with patch("hmm.inference_prep.get_selected_hmm_set", side_effect=ValueError("Boom")):
        with pytest.raises(ValueError, match="Boom"):
            filter_phrog_annotations("in.tsv", "out.tsv", "model.tsv")


def test_filter_dat_annotations_exception(temp_dir):
    """Hits the try-except block in filter_dat_annotations."""
    with patch("hmm.inference_prep.get_selected_hmm_set", side_effect=RuntimeError("Data Error")):
        with pytest.raises(RuntimeError, match="Data Error"):
            filter_dat_annotations("in.dat", "out.dat", "model.tsv")
