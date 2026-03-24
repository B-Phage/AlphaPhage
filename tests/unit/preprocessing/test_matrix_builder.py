import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

# Replace 'representation' with the actual package name if different
from representation.matrix_builder import build_all_v_all_matrix, main

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------


def create_mock_fastas(directory: Path, filenames: list):
    """Utility to create empty fasta files for testing."""
    directory.mkdir(parents=True, exist_ok=True)
    for name in filenames:
        (directory / f"{name}.fasta").touch()


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------


def test_build_all_v_all_matrix_success(tmp_path):
    """Verifies matrix is built correctly with sorted IDs and -1 values."""
    phage_dir = tmp_path / "phages"
    host_dir = tmp_path / "hosts"
    output_path = tmp_path / "output" / "matrix.tsv"

    # Create mock files (out of order to test sorting)
    create_mock_fastas(phage_dir, ["P2", "P1"])
    create_mock_fastas(host_dir, ["H2", "H1"])

    build_all_v_all_matrix(phage_dir, host_dir, output_path)

    # 1. Check if file exists
    assert output_path.exists()

    # 2. Verify contents
    df = pd.read_csv(output_path, sep="\t", index_col="host_id")

    # Check dimensions
    assert df.shape == (2, 2)
    # Check sorting
    assert list(df.index) == ["H1", "H2"]
    assert list(df.columns) == ["P1", "P2"]
    # Check values
    assert (df.values == -1).all()


def test_build_all_v_all_matrix_empty_dir_raises_error(tmp_path):
    """Verifies that FileNotFoundError is raised if a directory is empty."""
    phage_dir = tmp_path / "phages"
    host_dir = tmp_path / "hosts"
    output_path = tmp_path / "matrix.tsv"

    phage_dir.mkdir()
    host_dir.mkdir()
    # Create phages but NO hosts
    create_mock_fastas(phage_dir, ["P1"])

    with pytest.raises(FileNotFoundError, match="Could not find .fasta files"):
        build_all_v_all_matrix(phage_dir, host_dir, output_path)


def test_main_cli_entrypoint(tmp_path):
    """Tests the main() function by mocking CLI arguments."""
    phage_dir = tmp_path / "phages"
    host_dir = tmp_path / "hosts"
    output_file = tmp_path / "out.tsv"

    create_mock_fastas(phage_dir, ["P1"])
    create_mock_fastas(host_dir, ["H1"])

    # Mock sys.argv to simulate command line call
    test_args = [
        "matrix_builder.py",
        "--phage_dir",
        str(phage_dir),
        "--host_dir",
        str(host_dir),
        "--output_path",
        str(output_file),
    ]

    with patch.object(sys, "argv", test_args):
        main()

    assert output_file.exists()


def test_main_missing_args(tmp_path):
    """Tests that the CLI exits when required arguments are missing."""
    with patch.object(sys, "argv", ["matrix_builder.py"]):
        with pytest.raises(SystemExit):
            main()
