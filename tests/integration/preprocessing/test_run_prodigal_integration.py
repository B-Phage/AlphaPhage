"""
Run with:
conda activate preprocessing
export PYTHONPATH=$PYTHONPATH:$(pwd)/workflows:$(pwd)/packages
python -m unittest tests/integration/test_run_prodigal_integration.py
"""

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class TestRunProdigalIntegration(unittest.TestCase):

    def setUp(self):
        # Create temporary directories
        self.tempdir = Path(tempfile.mkdtemp())
        self.config_file = self.tempdir / "prodigal_test_config.yaml"
        self.input_dir = self.tempdir / "genomes"
        self.proteins_dir = self.tempdir / "proteins"

        self.input_dir.mkdir()
        self.proteins_dir.mkdir()

        # Create a tiny dummy genome
        self.genome_file = self.input_dir / "genome1.fasta"
        self.genome_file.write_text(">dummy\nATGCGTACGTA")

        # Create a minimal prodigal config
        self.config_file.write_text(
            f"""
prodigal_config:
  genome_dir: "{self.input_dir}"
  proteins_dir: "{self.proteins_dir}"
general:
  lst_genomes: []
  code_version: "integration-test-v1"
"""
        )

    def tearDown(self):
        # Remove temporary directory after test
        shutil.rmtree(self.tempdir)

    def test_run_prodigal_script_creates_outputs(self):
        """
        Run the actual run_prodigal.py script on the dummy genome
        and verify that the protein output is created.
        """

        script_path = Path("workflows/run_prodigal.py").resolve()
        if not script_path.exists():
            self.skipTest(f"{script_path} does not exist, skipping integration test")

        # Run the script using the same Python interpreter
        result = subprocess.run(
            [
                "coverage",
                "run",
                "-p",  # -p is crucial: it creates unique data files
                "--source=workflows",
                str(script_path),
                "--config",
                str(self.config_file),
                "--code_hash",
                str(script_path),
                "--force",
            ],
            capture_output=True,
            text=True,
            env={**os.environ, "COVERAGE_FILE": ".coverage.subs"},  # Direct data to a specific file
        )

        # Print stdout/stderr for debugging
        print(result.stdout)
        print(result.stderr)

        # Assert the script ran successfully
        self.assertEqual(result.returncode, 0, msg="run_prodigal.py failed")

        # Check that the output protein file exists
        expected_protein = self.proteins_dir / "genome1.fa"
        self.assertTrue(expected_protein.exists(), msg=f"{expected_protein} not created")
