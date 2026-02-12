import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def small_phrog_test_data(tmp_path):
    """Integration fixture: tiny FASTA, real HMM, and annotation CSV."""

    # FASTA file
    fasta_file = tmp_path / "proteins.fa"
    fasta_file.write_text(
        ">ECP1;NC_049926.1_13\nMKMPEKNDLLAAILAAKEQGIGAILAFAMAYLRGRYNGGAFTKTVIDATMCAIIAWFIRDLLGFAGLSSNLAYITSVFIGYIGTDSIGSLIKRFAAKKAGVEDGGNQ*\n>411_P1;411_P1_68\nMKMPEKHDLLAAILAAKEQGIGAILAFAMAYLRGRYNGGAFTKTVIDATMCAIIAWFIRDLLDFAGLSSNLAYITSVFIGYIGTDSIGSLIKRFAAKKAGIEDGGNQ*\n"
    )

    # Use real HMM file from data
    hmm_file = Path("tests/data/phrog_126.hmm")

    # Annotation CSV matching the HMM domain
    annotation_file = tmp_path / "annotations.csv"
    annotation_file.write_text("hmm_domain_name,annot\nPHROG_126,TestAnnotation\n")

    return {
        "fasta": fasta_file,
        "hmm": hmm_file,
        "annotation": annotation_file,
        "out_csv": tmp_path / "results.csv",
        "out_db": tmp_path / "results.duckdb",
    }


@pytest.fixture
def small_pfam_test_data(tmp_path):
    """Integration fixture: tiny FASTA, real HMM, and annotation CSV."""

    # Use real HMM file from data
    hmm_file = Path("tests/data/DUF1244.hmm")
    fasta_file = Path("tests/data/DUF1244.fa")

    # Annotation dat file matching the HMM domain
    annotation_file = Path("tests/data/Pfam-A_test.hmm.dat")

    return {
        "fasta": fasta_file,
        "hmm": hmm_file,
        "annotation": annotation_file,
        "out_csv": tmp_path / "results.csv",
        "out_db": tmp_path / "results.duckdb",
    }


def test_cli_runs_successfully_for_phrogs(small_phrog_test_data, tmp_path):
    """Integration test: run CLI on tiny test dataset using a real HMM file."""
    # fmt: off
    cmd = [
        sys.executable,
        "workflows/cli/hmm_scan.py",
        "-f", str(small_phrog_test_data["fasta"]),
        "-hp", str(small_phrog_test_data["hmm"]),
        "-a", str(small_phrog_test_data["annotation"]),
        "-m", "pfam",
        "-o", str(small_phrog_test_data["out_csv"]),
        "-db", str(small_phrog_test_data["out_db"]),
    ]
    # fmt: on

    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)

    # CLI should complete successfully
    assert result.returncode == 0

    # CSV results should exist and have expected columns
    assert small_phrog_test_data["out_csv"].exists()
    df = pd.read_csv(small_phrog_test_data["out_csv"])
    assert "n_rows" in df.columns
    assert "n_columns" in df.columns


def test_cli_runs_successfully_for_pfam(small_pfam_test_data, tmp_path):
    """Integration test: run CLI on tiny test dataset using a real HMM file."""
    # fmt: off
    cmd = [
        sys.executable,
        "workflows/cli/hmm_scan.py",
        "-f", str(small_pfam_test_data["fasta"]),
        "-hp", str(small_pfam_test_data["hmm"]),
        "-a", str(small_pfam_test_data["annotation"]),
        "-m", "pfam",
        "-o", str(small_pfam_test_data["out_csv"]),
        "-db", str(small_pfam_test_data["out_db"]),
    ]
    # fmt: off
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)

    # CLI should complete successfully
    assert result.returncode == 0

    # CSV results should exist and have expected columns
    assert small_pfam_test_data["out_csv"].exists()
    df = pd.read_csv(small_pfam_test_data["out_csv"])
    assert "n_rows" in df.columns
    assert "n_columns" in df.columns
