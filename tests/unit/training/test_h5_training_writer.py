import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
from data_contracts.h5_schema import H5DataKeys

from packages.representation.h5_writer import H5TrainingWriter


class TestH5TrainingWriter(unittest.TestCase):
    def setUp(self):
        """Initialize temporary directory and dummy data."""
        self.test_dir = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.test_dir.name)
        self.base_filename = "test_run"
        self.writer = H5TrainingWriter(self.output_dir, self.base_filename)
        self.keys = H5DataKeys()

        # Dummy data setup
        self.num_samples = 5
        self.data_dict = {
            "phage_vecs": np.random.randint(0, 10, (self.num_samples, 21), dtype=np.int32),
            "host_vecs": np.random.randint(0, 10, (self.num_samples, 28), dtype=np.int32),
            "labels": np.random.randint(0, 2, (self.num_samples,), dtype=np.int32),
            "pair_keys": np.array([["phage_1", "host_1"]] * self.num_samples).astype("S100"),
        }

        # Info metadata arrays (2 rows, N columns)
        self.phage_info = np.array([["p_id"] * 21, ["v1"] * 21]).astype("S100")
        self.host_info = np.array([["h_id"] * 28, ["v1"] * 28]).astype("S100")

    def tearDown(self):
        """Cleanup temporary files."""
        self.test_dir.cleanup()

    def test_file_naming_convention(self):
        """Verify the 'x_' and 'cl_' prefixes are correctly applied."""
        self.assertEqual(self.writer.x_path.name, f"x_{self.base_filename}.h5")
        self.assertEqual(self.writer.cl_path.name, f"cl_{self.base_filename}.h5")

    def test_write_creates_files(self):
        """Check if write() actually generates the physical files."""
        x_path, cl_path = self.writer.write(self.data_dict, self.phage_info, self.host_info)
        self.assertTrue(x_path.exists())
        self.assertTrue(cl_path.exists())

    def test_x_file_content_integrity(self):
        """Verify that the feature file contains all expected datasets and shapes."""
        x_path, _ = self.writer.write(self.data_dict, self.phage_info, self.host_info)

        with h5py.File(x_path, "r") as f:
            # Check for keys from Data Contract
            self.assertIn(self.keys.PHAGE_VEC, f)
            self.assertIn(self.keys.HOST_VEC, f)
            self.assertIn(self.keys.PAIR_INDICES, f)

            # Verify Shapes
            self.assertEqual(f[self.keys.PHAGE_VEC].shape, (self.num_samples, 21))
            self.assertEqual(f[self.keys.HOST_VEC].shape, (self.num_samples, 28))

            # Verify Indices (Identity mapping check [i, i])
            indices = f[self.keys.PAIR_INDICES][:]
            np.testing.assert_array_equal
