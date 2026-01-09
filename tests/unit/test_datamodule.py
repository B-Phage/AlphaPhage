import os
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from omegaconf import OmegaConf
from training.dataloaders.dataloaders import BinaryPairkeyDataModule, BinaryPairkeyFeatureDataset


class TestBinaryPairkeyDataModule(unittest.TestCase):
    def setUp(self):
        self.cfg = OmegaConf.create(
            {
                "seed": 42,
                # "verboselevel" : "ERROR",
                "debug_mode": False,
                # "train_features_path" : "../../bp/AlphaX/proc/sim_data/Tests/x_Test_50_50_TOK_test_AT_cond2_aligned_test_binary_pairkey.h5",
                # "train_labels_path" : "../../bp/AlphaX/proc/sim_data/Tests/cl_Test_50_50_TOK_test_AT_cond2_aligned_test_binary_pairkey.h5",
                "train_features_path": "tests/data/x_Test_50_50_TOK_test_AT_cond2_aligned_test_binary_pairkey.h5",
                "train_labels_path": "tests/data/cl_Test_50_50_TOK_test_AT_cond2_aligned_test_binary_pairkey.h5",
                # "train_features_path": "tests/data/x_Test_50_50_TOK_test_AT_cond2_unaligned_test_binary_pairkey.h5",
                # "train_labels_path": "tests/data/cl_Test_50_50_TOK_test_AT_cond2_unaligned_test_binary_pairkey.h5",
                # "debug_features_path" : "",
                # "debug_labels_path" : "",
                "pair_indices_name": "pair_indices",
                "dataset_name_host": "host_vec_1",
                "dataset_name_phage": "phage_vec_1",
                "reader_mode": "binary_pairkey",
                # "architecture" : "TransformerX",
                "num_workers": 0,
                "trainer": {
                    #     "logger_method": "MLFlowLogger",
                    #     "project": "AlphaPhage_Testing",
                    #     "run_name": "test_run_2",
                    #     "mlflow_tracking_uri": None,
                    #     "epochs": 300,
                    #     "accelerator": "cpu",
                    #     "devices": 1,
                    #     "precision": 32,
                    "batch_size": 10,
                    #     "save_top_k": 3,
                    #     "patience": 5,
                    "val_fraction": 0.15,
                },
                # "hyp": {
                #     "archspec": "nope_nt_nonormpostl_sum",
                #     "archspec2": "L4_bn",
                #     "dim": 8,
                #     "lcom": 30,
                #     "dim1": 8,
                #     "dim2": 8,
                #     "heads": 4,
                #     "mult": 4,
                #     "depth": 1,
                #     "vocab_size_out": 2,
                # },
                # "optimizer": {
                #     "lr": 0.001,
                # },
            }
        )

    def test_dataset_integrity(self):
        """Check if the dataset reads actual H5 data and returns correct types."""
        dataset = BinaryPairkeyFeatureDataset(
            self.cfg, self.cfg.train_features_path, self.cfg.train_labels_path
        )

        # Verify length is calculated from the file
        self.assertGreater(len(dataset), 0)

        # Pull a single sample
        x, y = dataset[0]

        # Check types and dimensions
        self.assertIsInstance(x, torch.Tensor)
        self.assertIsInstance(y, torch.Tensor)
        self.assertEqual(x.dtype, torch.long)
        # Ensure the feature vector is concatenated (host + phage)
        self.assertEqual(x.ndim, 1)

    def test_datamodule_splitting(self):
        """Verify the IndexBatchSampler correctly partitions the real file indices."""
        dm = BinaryPairkeyDataModule(self.cfg)
        dm.setup(stage="fit")

        train_loader = dm.train_dataloader()
        val_loader = dm.val_dataloader()

        # Verify the split ratio
        total_samples = len(dm.full_dataset)
        expected_val_size = int(total_samples * self.cfg.trainer.val_fraction)

        self.assertEqual(len(dm.val_idx), expected_val_size)

        # Ensure indices are unique and valid
        combined_indices = set(dm.train_idx).union(set(dm.val_idx))
        self.assertEqual(len(combined_indices), total_samples)

    def test_dataloader_batch_iteration(self):
        """Test the full BinaryPairkeyDataModule for stage = fit: reading from
        H5 via the DataLoader.
        """
        dm = BinaryPairkeyDataModule(self.cfg)
        dm.setup(stage="fit")
        train_loader = dm.train_dataloader()

        # Iterate through the first batch
        for i, (batch_x, batch_y) in enumerate(train_loader):
            self.assertEqual(batch_x.shape[0], self.cfg.trainer.batch_size)
            self.assertEqual(batch_y.ndim, 1)  # Batch of labels
            break  # One batch is enough for a unit test


if __name__ == "__main__":
    unittest.main()
