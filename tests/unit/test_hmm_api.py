from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from hmm.api import run_hmm_scan_on_file


@pytest.fixture
def dummy_fasta(tmp_path: Path):
    fasta = tmp_path / "dummy.fa"
    fasta.write_text(">protein1\nACDEFGHIKLMNPQRSTVWY\n")
    return fasta


@pytest.fixture
def dummy_hmm(tmp_path: Path):
    hmm_file = tmp_path / "dummy.hmm"
    hmm_file.write_text("FAKE_HMM_CONTENT")
    return hmm_file


@pytest.fixture
def dummy_annotation(tmp_path: Path):
    ann = tmp_path / "annotations.csv"
    ann.write_text("hmm_domain_name,description\nDUMMY,HMM dummy\n")
    return ann


@pytest.fixture
def dummy_out(tmp_path: Path):
    return tmp_path / "out.csv"


@pytest.fixture
def dummy_db(tmp_path: Path):
    return tmp_path / "out.duckdb"


# ---------------------------------------------
# Basic functionality
# ---------------------------------------------
@patch("hmm.api.load_hmms_from_file")
@patch("hmm.api.load_single_hmm_per_file")
@patch("hmm.api.run_hmm_search")
@patch("hmm.api.AnnotationProvider")
@patch("hmm.api.write_hmm_results_with_sequences")
@patch("hmm.api.write_hmm_scan_summary")
def test_basic_run(
    mock_write_summary,
    mock_write_db,
    mock_annotation,
    mock_run_hmm_search,
    mock_load_single,
    mock_load_file,
    dummy_fasta,
    dummy_hmm,
    dummy_annotation,
    dummy_out,
    dummy_db,
):
    # Setup mocks
    mock_load_file.return_value = ["HMM1"]
    mock_load_single.return_value = ["HMM1"]
    mock_run_hmm_search.return_value = MagicMock(__len__=lambda self: 1)
    mock_annotation.return_value = MagicMock()

    # Run function
    run_hmm_scan_on_file(
        in_fasta=dummy_fasta,
        hmm_path=dummy_hmm,
        annotation_file=dummy_annotation,
        method="phrogs",
        out_csv=dummy_out,
        out_database=dummy_db,
        evalue_cutoff=1e-3,
    )

    # Check that HMM loaders were called
    mock_load_file.assert_called_once()
    mock_annotation.assert_called_once_with("phrogs", dummy_annotation)
    mock_run_hmm_search.assert_called_once()
    mock_write_summary.assert_called_once()
    mock_write_db.assert_called_once()


# ---------------------------------------------
# HMM path does not exist
# ---------------------------------------------
def test_hmm_path_missing(dummy_fasta, dummy_annotation, dummy_out, dummy_db):
    fake_hmm = Path("/nonexistent/file.hmm")
    with pytest.raises(ValueError, match="HMM path does not exist"):
        run_hmm_scan_on_file(
            in_fasta=dummy_fasta,
            hmm_path=fake_hmm,
            annotation_file=dummy_annotation,
            method="phrogs",
            out_csv=dummy_out,
            out_database=dummy_db,
        )


# ---------------------------------------------
# HMM path is directory
# ---------------------------------------------
@patch("hmm.api.load_single_hmm_per_file")
@patch("hmm.api.run_hmm_search")
@patch("hmm.api.AnnotationProvider")
@patch("hmm.api.write_hmm_results_with_sequences")
@patch("hmm.api.write_hmm_scan_summary")
def test_hmm_path_directory(
    mock_write_summary,
    mock_write_db,
    mock_annotation,
    mock_run_hmm_search,
    mock_load_single,
    dummy_fasta,
    tmp_path,
    dummy_annotation,
    dummy_out,
    dummy_db,
):
    # Create directory of HMMs
    hmm_dir = tmp_path / "hmms"
    hmm_dir.mkdir()
    hmm_file = hmm_dir / "h1.hmm"
    hmm_file.write_text("FAKE_HMM")

    mock_load_single.return_value = ["HMM1"]
    mock_run_hmm_search.return_value = MagicMock(__len__=lambda self: 1)
    mock_annotation.return_value = MagicMock()

    run_hmm_scan_on_file(
        in_fasta=dummy_fasta,
        hmm_path=hmm_dir,
        annotation_file=dummy_annotation,
        method="phrogs",
        out_csv=dummy_out,
        out_database=dummy_db,
    )

    mock_load_single.assert_called_once_with(hmm_dir, "*.hmm")


# ---------------------------------------------
# No hits -> empty CSV
# ---------------------------------------------
@patch("hmm.api.load_hmms_from_file")
@patch("hmm.api.run_hmm_search")
@patch("hmm.api.AnnotationProvider")
@patch("hmm.api.write_hmm_results_with_sequences")
def test_no_hits_writes_empty_csv(
    mock_write_db,
    mock_annotation,
    mock_run_hmm_search,
    mock_load_file,
    dummy_fasta,
    dummy_hmm,
    dummy_annotation,
    tmp_path,
):
    out_csv = tmp_path / "out.csv"
    dummy_db = tmp_path / "out.duckdb"

    mock_load_file.return_value = ["HMM1"]
    # Simulate no hits
    mock_run_hmm_search.return_value = []

    mock_annotation.return_value = MagicMock()

    run_hmm_scan_on_file(
        in_fasta=dummy_fasta,
        hmm_path=dummy_hmm,
        annotation_file=dummy_annotation,
        method="phrogs",
        out_csv=out_csv,
        out_database=dummy_db,
    )

    assert out_csv.exists()
    assert out_csv.read_text() == ""


# ---------------------------------------------
# E-value cutoff respected
# ---------------------------------------------
@patch("hmm.api.load_hmms_from_file")
@patch("hmm.api.run_hmm_search")
@patch("hmm.api.AnnotationProvider")
@patch("hmm.api.write_hmm_results_with_sequences")
@patch("hmm.api.write_hmm_scan_summary")
def test_evalue_cutoff_passed(
    mock_write_summary,
    mock_write_db,
    mock_annotation,
    mock_run_hmm_search,
    mock_load_file,
    dummy_fasta,
    dummy_hmm,
    dummy_annotation,
    dummy_out,
    dummy_db,
):
    mock_load_file.return_value = ["HMM1"]
    mock_run_hmm_search.return_value = MagicMock(__len__=lambda self: 1)
    mock_annotation.return_value = MagicMock()

    cutoff = 1e-4
    run_hmm_scan_on_file(
        in_fasta=dummy_fasta,
        hmm_path=dummy_hmm,
        annotation_file=dummy_annotation,
        method="phrogs",
        out_csv=dummy_out,
        out_database=dummy_db,
        evalue_cutoff=cutoff,
    )

    # Ensure the run_hmm_search was called with the cutoff
    args, kwargs = mock_run_hmm_search.call_args
    assert kwargs["evalue_cutoff"] == cutoff
