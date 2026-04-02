import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
from data_contracts.h5_schema import H5DataKeys

from packages.representation.h5_writer import H5TrainingWriter


class TestH5TrainingWriter(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.test_dir.name)
        self.base_filename = "test_run"
        self.writer = H5TrainingWriter(self.output_dir, self.base_filename)
        self.keys = H5DataKeys()

        self.num_samples = 5
        # Generate dummy pair indices (e.g., [0,0], [1,1]...)
        self.pair_indices = np.stack([np.arange(self.num_samples)] * 2, axis=1).astype(np.int32)

        self.phage_vecs = np.random.randint(0, 10, (self.num_samples, 21), dtype=np.int32)
        self.host_vecs = np.random.randint(0, 10, (self.num_samples, 28), dtype=np.int32)
        self.labels = np.random.randint(0, 2, (self.num_samples,), dtype=np.int32)
        self.pair_keys = np.array([["phage_1", "host_1"]] * self.num_samples).astype("S100")

        self.phage_info = np.array([["p_id"] * 21, ["v1"] * 21]).astype("S100")
        self.host_info = np.array([["h_id"] * 28, ["v1"] * 28]).astype("S100")

    def tearDown(self):
        self.test_dir.cleanup()

    def test_write_creates_files(self):
        """Verify explicit argument passing creates physical files."""
        x_path, cl_path = self.writer.write(
            phage_vecs=self.phage_vecs,
            host_vecs=self.host_vecs,
            pair_indices=self.pair_indices,
            pair_keys=self.pair_keys,
            labels=self.labels,
            phage_info=self.phage_info,
            host_info=self.host_info,
        )
        self.assertTrue(x_path.exists())
        self.assertTrue(cl_path.exists())

    def test_x_file_content_integrity(self):
        """Verify HDF5 internal keys and shapes match the Data Contract."""
        x_path, _ = self.writer.write(
            phage_vecs=self.phage_vecs,
            host_vecs=self.host_vecs,
            pair_indices=self.pair_indices,
            pair_keys=self.pair_keys,
            labels=self.labels,
            phage_info=self.phage_info,
            host_info=self.host_info,
        )

        with h5py.File(x_path, "r") as f:
            self.assertIn(self.keys.PHAGE_VEC, f)
            self.assertIn(self.keys.HOST_VEC, f)
            self.assertEqual(f[self.keys.PHAGE_VEC].shape, (self.num_samples, 21))
            self.assertEqual(f[self.keys.HOST_VEC].shape, (self.num_samples, 28))
