import unittest
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

# Import the workflow logic
from workflows import run_kaptive3api


# Top-level stubs to ensure picklability for multiprocessing
def stub_worker(input_path, outputs, db_obj=None):
    """Stub worker that returns outputs without doing real work."""
    return outputs


def stub_output(input_path, output_dir=None):
    """Stub output function returning a dummy path."""
    return [Path("output.json")]


class TestKaptiveIntegration(unittest.TestCase):

    def setUp(self):
        """Set up dummy configuration objects."""
        self.mock_cfg = MagicMock()
        self.mock_cfg.genome_dir = "data/genomes"
        self.mock_cfg.kaptive_dir = "results/kaptive"
        self.mock_cfg.orgdb_full_path = "db/kaptive.db"
        self.mock_cfg.code_version = "1.0.0"
        self.mock_cfg.get.return_value = "integration_test"

        self.mock_run_cfg = MagicMock()
        self.mock_run_cfg.n_jobs = 1

        self.mock_gen_cfg = MagicMock()
        self.mock_gen_cfg.lst_genomes = ["sample1"]
        self.mock_gen_cfg.code_version = "v1"

    @patch("run_kaptive3api.Orchestrator")
    @patch("run_kaptive3api.load_database")
    @patch("run_kaptive3api.get_config")
    @patch("run_kaptive3api.compute_combined_code_version")
    @patch("run_kaptive3api.os.makedirs")
    @patch("orchestration.orchestrator.hash_file")
    def test_workflow_orchestration(
        self, mock_hash_file, mock_makedirs, mock_hash, mock_get_config, mock_load_db, MockOrchClass
    ):
        """
        Integration Test: Verify workflow initialization with picklable stubs.
        """
        # 1. Setup Mock Returns
        mock_hash_file.return_value = "dummy_file_hash"
        mock_get_config.side_effect = [self.mock_run_cfg, self.mock_gen_cfg, self.mock_cfg]
        mock_hash.return_value = "combined_hash_123"
        mock_load_db.return_value = MagicMock(name="KaptiveDB")

        # 2. Trigger the instantiation
        genome_files = [Path(self.mock_cfg.genome_dir) / "sample1.fasta"]

        # Use the mocked class reference directly to ensure the mock is triggered
        orch = MockOrchClass(
            workflow_name="kapsid_integration_test",
            hash_store_dir=Path("ml_assets/data/workflow/hashes"),
            logger_name="kaptive",
        )

        # Define the instance as the return value of the class mock
        orch_instance = MockOrchClass.return_value

        orch.run(
            inputs=genome_files,
            output_func=stub_output,
            worker=stub_worker,
            code_id="combined_hash_123",
            parallelism=1,
            force=False,
        )

        # 3. Assertions
        # Verify the class was instantiated
        self.assertEqual(MockOrchClass.call_count, 1)

        # Verify .run() was called on the instance
        orch_instance.run.assert_called_once()

        # Verify specific parameters passed to run
        actual_code_id = orch_instance.run.call_args[1]["code_id"]
        self.assertEqual(actual_code_id, "combined_hash_123")

    @patch("kaptive3api.kaptive3api_functions.typing_pipeline")
    def test_run_kapsid_integration(self, mock_pipeline):
        """Verify data flow inside the worker function run_kapsid."""
        from kaptive3api.kaptive3api_functions import run_kapsid

        mock_result = MagicMock()
        mock_result.format.return_value = '{"sample_name": "s1", "best_match": "K1", "confidence": "High", "phenotype": "K1", "problems": "None", "percent_identity": 100, "percent_coverage": 100}'
        mock_pipeline.return_value = mock_result

        mock_db = MagicMock()
        input_path = Path("test.fasta")
        output_paths = [Path("test.json")]

        with patch("builtins.open", mock_open()) as mocked_file:
            run_kapsid(input_path, output_paths, mock_db)
            mock_pipeline.assert_called_with(input_path, mock_db)
            mocked_file.assert_called_with(Path("test.json"), "w")


if __name__ == "__main__":
    unittest.main()
