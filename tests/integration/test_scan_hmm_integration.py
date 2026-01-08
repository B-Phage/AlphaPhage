import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def small_test_data(tmp_path):
    """Integration fixture: tiny FASTA, real HMM, and annotation CSV."""

    # FASTA file
    fasta_file = tmp_path / "proteins.faa"
    fasta_file.write_text(
        ">ECP1;NC_049926.1_13\nMKMPEKNDLLAAILAAKEQGIGAILAFAMAYLRGRYNGGAFTKTVIDATMCAIIAWFIRDLLGFAGLSSNLAYITSVFIGYIGTDSIGSLIKRFAAKKAGVEDGGNQ*\n>411_P1;411_P1_68\nMKMPEKHDLLAAILAAKEQGIGAILAFAMAYLRGRYNGGAFTKTVIDATMCAIIAWFIRDLLDFAGLSSNLAYITSVFIGYIGTDSIGSLIKRFAAKKAGIEDGGNQ*\n"
    )

    # Use real HMM file from fixtures
    hmm_file = Path("tests/fixtures/phrog_126.hmm")

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


def test_cli_runs_successfully(small_test_data, tmp_path):
    """Integration test: run CLI on tiny test dataset using a real HMM file."""
    cmd = [
        sys.executable,
        "workflows/cli/hmm_scan.py",
        "-f",
        str(small_test_data["fasta"]),
        "-hp",
        str(small_test_data["hmm"]),
        "-a",
        str(small_test_data["annotation"]),
        "-m",
        "phrogs",
        "-o",
        str(small_test_data["out_csv"]),
        "-db",
        str(small_test_data["out_db"]),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)

    # CLI should complete successfully
    assert result.returncode == 0

    # CSV results should exist and have expected columns
    assert small_test_data["out_csv"].exists()
    df = pd.read_csv(small_test_data["out_csv"])
    assert "protein_name" in df.columns
    assert "hmm_domain_name" in df.columns
