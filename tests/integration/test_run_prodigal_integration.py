"""
Run with:
conda activate preprocessing
PYTHONPATH=./packages pytest tests/integration/ --cov=preprocessing --cov-report=html
"""

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
        self.output_dir = self.tempdir / "proteins"

        self.input_dir.mkdir()
        self.output_dir.mkdir()

        # Create a tiny dummy genome
        self.genome_file = self.input_dir / "genome1.fasta"
        self.genome_file.write_text(">dummy\nATGCGTACGTA")

        # Create a minimal prodigal config
        self.config_file.write_text(
            f"""
prodigal_config:
  genome_dir: "{self.input_dir}"
  output_dir: "{self.output_dir}"
  lst_genomes: []
  n_jobs: 1
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
            [sys.executable, str(script_path), "--config", str(self.config_file), "--force"],
            capture_output=True,
            text=True,
        )

        # Print stdout/stderr for debugging
        print(result.stdout)
        print(result.stderr)

        # Assert the script ran successfully
        self.assertEqual(result.returncode, 0, msg="run_prodigal.py failed")

        # Check that the output protein file exists
        expected_protein = self.output_dir / "genome1.faa"
        self.assertTrue(expected_protein.exists(), msg=f"{expected_protein} not created")
