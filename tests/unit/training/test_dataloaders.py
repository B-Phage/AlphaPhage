import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import h5py
import numpy as np
import torch
from dataloaders.dataloaders import (
    BinaryPairkeyDataModule,
    BinaryPairkeyFeatureDataset,
    IndexBatchSampler,
)
from omegaconf import OmegaConf


#####################
class TestBinaryPairkeyDataModule(unittest.TestCase):
    def setUp(self):
        # 1. Create a temporary directory for H5 files
        self.test_dir = tempfile.TemporaryDirectory()
        self.feat_path = os.path.join(self.test_dir.name, "features.h5")
        self.lab_path = os.path.join(self.test_dir.name, "labels.h5")

        # 2. Create dummy H5 data
        with h5py.File(self.feat_path, "w") as f:
            f.create_dataset("host_vec_1", data=np.random.randint(0, 20, (10, 128)))
            f.create_dataset("phage_vec_1", data=np.random.randint(0, 20, (10, 128)))
            # Create 10 pairs (diagonal for simplicity)
            f.create_dataset("pair_indices", data=np.array([[i, i] for i in range(10)]))

        with h5py.File(self.lab_path, "w") as f:
            f.create_dataset("data", data=np.random.randint(0, 2, (10,)))

        # 3. Setup Config pointing to these temp files
        self.cfg = OmegaConf.create(
            {
                "seed": 42,
                "debug_mode": False,
                "train_features_path": self.feat_path,
                "train_labels_path": self.lab_path,
                "debug_features_path": self.feat_path,
                "debug_labels_path": self.lab_path,
                "pair_indices_name": "pair_indices",
                "dataset_name_host": "host_vec_1",
                "dataset_name_phage": "phage_vec_1",
                "num_workers": 0,
                "batch_size": 2,
                "val_fraction": 0.2,
            }
        )

    def tearDown(self):
        self.test_dir.cleanup()

    def test_dataset_loading(self):
        """Verifies that the dataset reads the temp H5 files correctly."""
        dataset = BinaryPairkeyFeatureDataset(self.cfg, self.feat_path, self.lab_path)
        self.assertEqual(len(dataset), 10)

        x, y = dataset[0]
        # Check concatenation: 128 (host) + 128 (phage) = 256
        self.assertEqual(x.shape[0], 256)
        self.assertIsInstance(x, torch.LongTensor)

    def test_datamodule_full_flow(self):
        """Tests the DataModule setup and dataloader iteration."""
        dm = BinaryPairkeyDataModule(self.cfg)
        dm.setup(stage="fit")

        loader = dm.train_dataloader()
        batch_x, batch_y = next(iter(loader))

        # Batch size is 2
        self.assertEqual(batch_x.shape[0], 2)
        self.assertEqual(batch_x.shape[1], 256)

    def test_index_batch_sampler_logic(self):
        """Tests the sampler independently for 'all' and normal modes."""
        indices = list(range(10))
        # Test normal batching
        sampler = IndexBatchSampler(indices, batch_size=3)
        self.assertEqual(len(sampler), 4)  # 3, 3, 3, 1 -> 4 batches

        # Test 'all' mode
        sampler_all = IndexBatchSampler(indices, batch_size="all")
        self.assertEqual(len(sampler_all), 1)

    def test_sampler_all_and_invalid_logic(self):
        """Directly forces the 'all' branch and the ValueError branch."""
        indices = [1, 2, 3]

        # Hits the 'elif batch_size == "all"' block
        sampler_all = IndexBatchSampler(indices, batch_size="all")
        self.assertEqual(sampler_all.num_batches, 1)

        # Hits the 'else: raise ValueError' block (Lines 50-51)
        with self.assertRaises(ValueError):
            IndexBatchSampler(indices, batch_size=None)

    def test_index_batch_sampler_invalid_type(self):
        """Covers lines 46-51: batch_size not type int or 'all'."""
        indices = list(range(10))
        # Test with an invalid type (float)
        with self.assertRaisesRegex(ValueError, "batch_size not type int or 'all'"):
            IndexBatchSampler(indices, batch_size=3.14)

        # Test with an invalid string
        with self.assertRaisesRegex(ValueError, "batch_size not type int or 'all'"):
            IndexBatchSampler(indices, batch_size="some_string")

    def test_datamodule_stages(self):
        """Ensure all setup stages work without error."""
        dm = BinaryPairkeyDataModule(self.cfg)
        for stage in ["fit", "test", "predict"]:
            with self.subTest(stage=stage):
                dm.setup(stage=stage)
                if stage == "test":
                    self.assertTrue(hasattr(dm, "test_dataset"))

    def test_datamodule_predict_stage(self):
        """Covers lines 218-219: stage == 'predict' in setup()."""
        dm = BinaryPairkeyDataModule(self.cfg)

        # Explicitly call the predict stage
        dm.setup(stage="predict")

        # Verify the predict_dataset was created
        self.assertTrue(hasattr(dm, "predict_dataset"))
        self.assertIsInstance(dm.predict_dataset, BinaryPairkeyFeatureDataset)
        self.assertEqual(len(dm.predict_dataset), 10)

    def test_datamodule_debug_mode(self):
        """Hits the debug_mode branch in setup()."""
        self.cfg.debug_mode = True
        dm = BinaryPairkeyDataModule(self.cfg)
        dm.setup(stage="fit")
        self.assertEqual(dm.full_dataset.features_path, self.feat_path)

    def test_datamodule_debug_mode_paths(self):
        """Verifies that debug_mode correctly switches H5 file paths."""
        # 1. Setup a specific debug path in config
        debug_feat = os.path.join(self.test_dir.name, "debug_feat.h5")
        debug_lab = os.path.join(self.test_dir.name, "debug_lab.h5")

        # Create the dummy debug files so initialization doesn't fail
        for path in [debug_feat, debug_lab]:
            with h5py.File(path, "w") as f:
                f.create_dataset("data", data=np.zeros(5))
                f.create_dataset("host_vec_1", data=np.zeros((5, 128)))
                f.create_dataset("phage_vec_1", data=np.zeros((5, 128)))
                f.create_dataset("pair_indices", data=np.array([[0, 0]]))

        self.cfg.debug_mode = True
        self.cfg.debug_features_path = debug_feat
        self.cfg.debug_labels_path = debug_lab

        dm = BinaryPairkeyDataModule(self.cfg)
        dm.setup(stage="fit")

        # Verify the internal dataset is pointing to the debug path, not the train path
        self.assertEqual(dm.full_dataset.features_path, debug_feat)
        self.assertEqual(dm.full_dataset.labels_path, debug_lab)

    def test_dataset_lazy_loading_already_open(self):
        """Ensure the code that skips re-opening H5 files is covered."""
        dataset = BinaryPairkeyFeatureDataset(self.cfg, self.feat_path, self.lab_path)
        _ = dataset[0]  # First hit: opens files
        _ = dataset[1]  # Second hit: skips hasattr block
        self.assertTrue(hasattr(dataset, "Ynp"))

    def test_val_dataloader_execution(self):
        """Forces execution of line 256 by calling val_dataloader()."""
        dm = BinaryPairkeyDataModule(self.cfg)
        dm.setup(stage="fit")

        # This line triggers the return statement on line 256
        v_loader = dm.val_dataloader()

        # Optional: verify it is actually a DataLoader
        self.assertIsInstance(v_loader, torch.utils.data.DataLoader)
        self.assertEqual(len(dm.val_idx), 2)  # Based on your setup of 10 samples & 0.2 fraction

    def test_predict_dataloader_execution(self):
        """Verifies the predict_dataloader returns a valid loader and handles inference settings."""
        # 1. We must add the predict keys to cfg because the DataModule
        # specifically looks for them in the 'predict' stage.
        self.cfg.predict_features_path = self.feat_path
        self.cfg.predict_labels_path = self.lab_path

        dm = BinaryPairkeyDataModule(self.cfg)

        # 2. Setup the predict stage specifically
        dm.setup(stage="predict")

        # 3. Call the predict_dataloader
        p_loader = dm.predict_dataloader()

        # 4. Assertions
        self.assertIsInstance(p_loader, torch.utils.data.DataLoader)
        self.assertEqual(p_loader.batch_size, self.cfg.batch_size)

        # Verify we can actually get data out of it
        batch_x, batch_y = next(iter(p_loader))
        self.assertEqual(batch_x.shape[0], self.cfg.batch_size)

        # Verify shuffle is False (by checking the sampler type or property)
        # In PyTorch, a non-shuffled DataLoader uses a SequentialSampler
        self.assertIsInstance(p_loader.sampler, torch.utils.data.SequentialSampler)


if __name__ == "__main__":
    unittest.main()
