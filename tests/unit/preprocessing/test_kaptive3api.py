import json
import os
import unittest
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

from omegaconf import OmegaConf
from run_kaptive3api import kapsid_output_func, run_kapsid


class TestKaptiveRunner(unittest.TestCase):

    def setUp(self):
        """Set up mock configurations and expected data schemas."""
        self.input_path = Path("data/raw/sample_01.fasta")
        self.output_paths = [Path("data/processed/sample_01.json")]
        self.mock_db = MagicMock(name="KaptiveDatabase")

        # The "Truth" data we expect the Kaptive library to return
        self.kaptive_raw_json = {
            "sample_name": "sample_01",
            "best_match": "K12",
            "confidence": "High",
            "phenotype": "K-type 12",
            "problems": "None",
            "percent_identity": 99.8,
            "percent_coverage": 100.0,
            "ignored_metadata": "some_heavy_string_we_dont_need",
        }

    # --- 1. Data Schema & Transformation Tests ---

    @patch("run_kaptive3api.typing_pipeline")
    def test_run_kapsid_data_extraction(self, mock_pipeline):
        """MLOps: Verify the worker extracts only the 7 required features."""
        # 1. Setup Mock Result Object
        mock_result = MagicMock()
        mock_result.format.return_value = json.dumps(self.kaptive_raw_json)
        mock_pipeline.return_value = mock_result

        # 2. Mock File I/O
        m = mock_open()
        with patch("builtins.open", m):
            run_kapsid(self.input_path, self.output_paths, self.mock_db)

        # 3. Capture and Verify Written Data
        handle = m()
        # Collect all calls to write()
        written_content = "".join(call.args[0] for call in handle.write.call_args_list)
        written_json = json.loads(written_content)

        # Check for specific keys required for downstream ML
        expected_keys = [
            "sample_name",
            "best_match",
            "confidence",
            "phenotype",
            "problems",
            "percent_identity",
            "percent_coverage",
        ]

        for key in expected_keys:
            with self.subTest(key=key):
                self.assertIn(key, written_json)

        # Ensure 'ignored_metadata' was stripped out to save storage/memory
        self.assertNotIn("ignored_metadata", written_json)
        self.assertEqual(len(written_json), 7)

    # --- 2. Type Safety & Guardrail Tests ---

    @patch("run_kaptive3api.typing_pipeline")
    def test_run_kapsid_invalid_return_type(self, mock_pipeline):
        """MLOps: Ensure the pipeline fails if the library returns a non-string."""
        mock_result = MagicMock()
        # Simulate a bug where the library returns a list instead of a JSON string
        mock_result.format.return_value = ["sample_name", "K1"]
        mock_pipeline.return_value = mock_result

        with self.assertRaises(TypeError) as cm:
            run_kapsid(self.input_path, self.output_paths, self.mock_db)

        self.assertIn("Expected JSON string", str(cm.exception))

    # --- 4. Success Branch (Full Flow Mock) ---
    @patch("run_kaptive3api.typing_pipeline")
    def test_run_kapsid_handles_none_result(self, mock_pipeline):
        """Ensure the script handles 'None' alignment results gracefully."""
        mock_pipeline.return_value = None

        m = mock_open()
        with patch("builtins.open", m):
            run_kapsid(self.input_path, self.output_paths, self.mock_db)

        handle = m()
        written_content = "".join(call.args[0] for call in handle.write.call_args_list)
        written_json = json.loads(written_content)

        # Should be an empty dict if result is None
        self.assertEqual(written_json, {})


class TestKaptiveOutputLogic(unittest.TestCase):

    def test_kapsid_output_func_path_construction(self):
        """
        MLOps Critical: Verify that the output path is correctly derived from
        the input filename and the specified output directory.
        """
        # 1. Define inputs
        input_path = Path("data/raw/genomes/sample_xyz.fasta")
        output_dir = Path("ml_assets/results/kaptive_runs")

        # 2. Call the function
        result = kapsid_output_func(input_path, output_dir)

        # 3. Expected result
        # Stem of sample_xyz.fasta is sample_xyz
        expected_file = output_dir / "sample_xyz.json"

        # 4. Assertions
        self.assertIsInstance(result, list, "Orchestrator expects a list of paths.")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], expected_file)

    def test_kapsid_output_func_handles_complex_extensions(self):
        """Ensure .fasta.gz or other suffixes don't break the naming convention."""
        input_path = Path("/tmp/bacteria.fna.gz")
        output_dir = Path("out")

        result = kapsid_output_func(input_path, output_dir)

        # .stem on bacteria.fna.gz returns bacteria.fna
        # If you want it to be just 'bacteria', you'd use .name.split('.')[0]
        # But based on your current code using .stem:
        expected_file = Path("out/bacteria.fna.json")
        self.assertEqual(result[0], expected_file)


if __name__ == "__main__":
    unittest.main()
