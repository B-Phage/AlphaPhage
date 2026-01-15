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

# class TestBinaryPairkeyMock_h5(unittest.TestCase):
#     def setUp(self):
#         self.cfg = OmegaConf.create(
#             {
#                 "seed": 42,
#                 "debug_mode": False,
#                 "train_features_path": "tests/data/x_Test_50_50_TOK_test_AT_cond2_aligned_test_binary_pairkey.h5",
#                 "train_labels_path": "tests/data/cl_Test_50_50_TOK_test_AT_cond2_aligned_test_binary_pairkey.h5",
#                 "debug_features_path": "tests/data/x_Test_50_50_TOK_test_AT_cond2_aligned_test_binary_pairkey.h5",
#                 "debug_labels_path": "tests/data/cl_Test_50_50_TOK_test_AT_cond2_unaligned_test_binary_pairkey.h5",
#                 "pair_indices_name": "pair_indices",
#                 "dataset_name_host": "host_vec_1",
#                 "dataset_name_phage": "phage_vec_1",
#                 "reader_mode": "binary_pairkey",
#                 "num_workers": 0,
#                 "trainer": {
#                     "batch_size": 10,
#                     "val_fraction": 0.15,
#                 },
#             }
#         )
#         # Create a temp file that behaves exactly like your production data
#         self.test_dir = tempfile.TemporaryDirectory()
#         self.feat_path = os.path.join(self.test_dir.name, "feat.h5")
#         self.lab_path = os.path.join(self.test_dir.name, "lab.h5")

#         # Populate with tiny dummy data
#         with h5py.File(self.feat_path, "w") as f:
#             f.create_dataset("host_vec_1", data=np.random.randint(0, 10, (5, 128)))
#             f.create_dataset("phage_vec_1", data=np.random.randint(0, 10, (5, 128)))
#             f.create_dataset("pair_indices", data=np.array([[0, 0], [1, 1]]))

#         with h5py.File(self.lab_path, "w") as f:
#             f.create_dataset("data", data=np.array([1, 0]))

#     def test_real_logic(self):
#         # Now you test the REAL class with NO mock flags
#         dataset = BinaryPairkeyFeatureDataset(self.cfg, self.feat_path, self.lab_path)
#         x, y = dataset[0]
#         self.assertEqual(len(x), 256)  # 128 + 128


# class TestBinaryPairkeyDataModule(unittest.TestCase):
#     def setUp(self):
#         self.cfg = OmegaConf.create(
#             {
#                 "seed": 42,
#                 "debug_mode": False,
#                 "train_features_path": "tests/data/x_Test_50_50_TOK_test_AT_cond2_aligned_test_binary_pairkey.h5",
#                 "train_labels_path": "tests/data/cl_Test_50_50_TOK_test_AT_cond2_aligned_test_binary_pairkey.h5",
#                 "debug_features_path": "tests/data/x_Test_50_50_TOK_test_AT_cond2_aligned_test_binary_pairkey.h5",
#                 "debug_labels_path": "tests/data/cl_Test_50_50_TOK_test_AT_cond2_unaligned_test_binary_pairkey.h5",
#                 "pair_indices_name": "pair_indices",
#                 "dataset_name_host": "host_vec_1",
#                 "dataset_name_phage": "phage_vec_1",
#                 "reader_mode": "binary_pairkey",
#                 "num_workers": 0,
#                 "trainer": {
#                     "batch_size": 10,
#                     "val_fraction": 0.15,
#                 },
#             }
#         )

#     ## 2. Test BatchSampler with 'all' string
#     def test_index_batch_sampler_all(self):
#         """Tests the 'all' batch size branch in IndexBatchSampler."""
#         indices = list(range(100))
#         sampler = IndexBatchSampler(indices, batch_size="all")
#         self.assertEqual(sampler.num_batches, 1)
#         self.assertEqual(len(next(iter(sampler))), 100)

#     ## 3. Test setup stages: 'test' and 'predict'
#     def test_datamodule_stages(self):
#         """Triggers the 'test' and 'predict' branches in setup()."""
#         dm = BinaryPairkeyDataModule(self.cfg)

#         dm.setup(stage="test")
#         self.assertIsNotNone(dm.test_dataset)

#         dm.setup(stage="predict")
#         self.assertIsNotNone(dm.predict_dataset)

#     ## 4. Test Debug Mode paths
#     def test_datamodule_debug_mode(self):
#         """Triggers the debug_mode branch in setup()."""
#         self.cfg.debug_mode = True
#         dm = BinaryPairkeyDataModule(self.cfg)
#         dm.setup(stage="fit")
#         # If this runs without FileNotFoundError, the path switching logic worked
#         self.assertTrue(dm.full_dataset.features_path == self.cfg.debug_features_path)

#     ## 5. Test Sampler Iteration and Permutation
#     def test_sampler_permutation(self):
#         """Ensures the permutate/shuffle logic is called."""
#         indices = [0, 1, 2, 3, 4]
#         sampler = IndexBatchSampler(indices, batch_size=2)
#         first_pass = list(iter(sampler))
#         second_pass = list(iter(sampler))
#         # It's statistically possible but unlikely to be identical if shuffled
#         self.assertEqual(len(first_pass), 3)  # 2+2+1

#     def test_dataset_integrity(self):
#         """Check if the dataset reads actual H5 data and returns correct types."""
#         dataset = BinaryPairkeyFeatureDataset(
#             self.cfg, self.cfg.train_features_path, self.cfg.train_labels_path
#         )

#         # Verify length is calculated from the file
#         self.assertGreater(len(dataset), 0)

#         # Pull a single sample
#         x, y = dataset[0]

#         # Check types and dimensions
#         self.assertIsInstance(x, torch.Tensor)
#         self.assertIsInstance(y, torch.Tensor)
#         self.assertEqual(x.dtype, torch.long)
#         # Ensure the feature vector is concatenated (host + phage)
#         self.assertEqual(x.ndim, 1)

#     def test_datamodule_splitting(self):
#         """Verify the IndexBatchSampler correctly partitions the real file indices."""
#         dm = BinaryPairkeyDataModule(self.cfg)
#         dm.setup(stage="fit")

#         train_loader = dm.train_dataloader()
#         val_loader = dm.val_dataloader()

#         # Verify the split ratio
#         total_samples = len(dm.full_dataset)
#         expected_val_size = int(total_samples * self.cfg.trainer.val_fraction)

#         self.assertEqual(len(dm.val_idx), expected_val_size)

#         # Ensure indices are unique and valid
#         combined_indices = set(dm.train_idx).union(set(dm.val_idx))
#         self.assertEqual(len(combined_indices), total_samples)

#     def test_dataloader_batch_iteration(self):
#         """Test the full BinaryPairkeyDataModule for stage = fit: reading from
#         H5 via the DataLoader.
#         """
#         dm = BinaryPairkeyDataModule(self.cfg)
#         dm.setup(stage="fit")
#         train_loader = dm.train_dataloader()

#         # Iterate through the first batch
#         for i, (batch_x, batch_y) in enumerate(train_loader):
#             self.assertEqual(batch_x.shape[0], self.cfg.trainer.batch_size)
#             self.assertEqual(batch_y.ndim, 1)  # Batch of labels
#             break  # One batch is enough for a unit test


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
                "trainer": {
                    "batch_size": 2,
                    "val_fraction": 0.2,
                },
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

    def test_datamodule_stages(self):
        """Ensure all setup stages work without error."""
        dm = BinaryPairkeyDataModule(self.cfg)
        for stage in ["fit", "test", "predict"]:
            with self.subTest(stage=stage):
                dm.setup(stage=stage)
                if stage == "test":
                    self.assertTrue(hasattr(dm, "test_dataset"))


if __name__ == "__main__":
    unittest.main()
