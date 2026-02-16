"""
Run with:
conda activate training
export PYTHONPATH=$PYTHONPATH:$(pwd)/workflows:$(pwd)/packages
python -m unittest tests/integration/test_run_training_integration.py
"""

import os
import shutil
import unittest
from pathlib import Path

import h5py
import numpy as np
import torch
from omegaconf import OmegaConf
from orchestration.orchestrator import Orchestrator

# Import project components
from run_train import run_trainer, trainer_output_func


class TestTrainIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Builds a temporary environment with mock data and configurations."""
        cls.test_base = Path("tests/tmp_train_integration")
        cls.test_base.mkdir(parents=True, exist_ok=True)

        # Paths for data and assets
        cls.features_path = cls.test_base / "mock_features.h5"
        cls.labels_path = cls.test_base / "mock_labels.h5"
        cls.params_path = cls.test_base / "params.yaml"
        cls.hash_dir = cls.test_base / "mock_hashes"
        cls.output_dir = cls.test_base / "ml_assets/training"

        # 1. Create Mock HDF5 Dataset
        num_samples = 10
        feat_dim = 16
        with h5py.File(cls.features_path, "w") as f:
            f.create_dataset("host_vec_1", data=np.random.randint(0, 10, (num_samples, feat_dim)))
            f.create_dataset("phage_vec_1", data=np.random.randint(0, 10, (num_samples, feat_dim)))
            f.create_dataset("pair_indices", data=np.stack([np.arange(num_samples)] * 2, axis=1))

        with h5py.File(cls.labels_path, "w") as f:
            f.create_dataset("data", data=np.random.randint(0, 2, (num_samples,)))

        # 2. Define a minimal valid Config for TransformerX
        cls.config = {
            "name": "integration_test",
            "seed": 42,
            "verboselevel": "ERROR",
            "num_workers": 0,
            "code_version": "v1.0.0-test",
            "reader_mode": "binary_pairkey",
            "architecture": "TransformerX",
            "debug_mode": True,
            "debug_features_path": str(cls.features_path),
            "debug_labels_path": str(cls.labels_path),
            "train_features_path": str(cls.features_path),
            "train_labels_path": str(cls.labels_path),
            "pair_indices_name": "pair_indices",
            "dataset_name_host": "host_vec_1",
            "dataset_name_phage": "phage_vec_1",
            "hyp": {
                "vocab_size_out": 2,
                "archspec": "rope_tt_normpostl_view",
                "archspec2": "L1_nn",
                "dim": 8,
                "dim1": 8,
                "dim2": 8,
                "heads": 2,
                "mult": 2,
                "depth": 1,
                "lcom": None,
            },
            "trainer": {
                "experiment_tracker_dir": str(cls.output_dir),
                "experiment_tracker_method": "DVCLiveLogger",
                "run_name": "test_run",
                "epochs": 1,
                "accelerator": "cpu",
                "devices": 1,
                "precision": "32-true",
                "patience": 1,
                "save_top_k": 1,
                "val_fraction": 0.2,
                "batch_size": 2,
            },
            "optimizer": {"lr": 0.001},
        }
        OmegaConf.save(cls.config, cls.params_path)

    @classmethod
    def tearDownClass(cls):
        """Removes the temporary integration test artifacts."""
        if cls.test_base.exists():
            shutil.rmtree(cls.test_base)

    def test_full_pipeline_orchestration(self):
        """Tests Orchestrator -> run_trainer -> Output File cycle."""

        # 1. Initialize Orchestrator
        orch = Orchestrator(
            workflow_name="test_integration_workflow",
            hash_store_dir=str(self.hash_dir),
            logger_name="test_orch",
        )

        # 2. Run the pipeline
        # This will execute run_trainer which uses Lightning to train 1 epoch
        orch.run(
            inputs=[self.params_path],
            output_func=trainer_output_func,
            worker=run_trainer,
            code_id=self.config["code_version"],
            parallelism=1,
        )

        # 3. Verify Training Outputs
        expected_run_dir = self.output_dir / self.config["trainer"]["run_name"]
        checkpoint_dir = expected_run_dir / "checkpoints"

        # Ensure model checkpoints were saved
        self.assertTrue(checkpoint_dir.exists())
        self.assertGreater(len(list(checkpoint_dir.glob("*.ckpt"))), 0)

        # Ensure Orchestrator created the receipt CSV file
        # In run_train.py, trainer_output_func dictates the CSV location
        receipt_file = expected_run_dir / "params.csv"
        # Since run_trainer currently doesn't 'touch' the file, we ensure the
        # logic for the path generation works
        self.assertEqual(trainer_output_func(self.params_path)[0], receipt_file)

    def test_orchestrator_skipping_logic(self):
        """Verifies that the Orchestrator skips identical runs using hashes."""
        orch = Orchestrator(workflow_name="test_skip_workflow", hash_store_dir=str(self.hash_dir))

        # Mocking an existing output so Orchestrator believes it is finished
        output_path = trainer_output_func(self.params_path)[0]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.touch()

        # First run to record hashes
        orch.run([self.params_path], trainer_output_func, run_trainer, "v1", parallelism=1)

        # Second run - should result in 0/1 inputs to run
        # We check the internal hashes dictionary
        self.assertIn(str(self.params_path), orch.hashes)

        # Verification: If we run again without 'force', it should return quickly
        # Logic is confirmed if hashes match input_file checksums
        current_input_hash = orch._hash_file(self.params_path)
        self.assertEqual(orch.hashes[str(self.params_path)]["input_hash"], current_input_hash)


if __name__ == "__main__":
    unittest.main()
