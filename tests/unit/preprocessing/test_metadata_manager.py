import unittest

import numpy as np
import pandas as pd

from packages.representation.metadata_manager import MetadataManager


class TestMetadataManager(unittest.TestCase):
    def setUp(self):
        """Set up a sample metadata dataframe."""
        self.meta_df = pd.DataFrame(
            {
                "hmm_id": ["phrog_1", "phrog_2", "KL10"],
                "clustering_version": ["v1.0", "v2.1", "v1.0"],
            }
        )
        self.cols = ["phrog_1", "phrog_2", "phrog_99"]

    def test_build_info_array_basic(self):
        """Test standard HMM to version mapping without overrides."""
        result = MetadataManager.build_info_array(self.cols, self.meta_df)

        # Verify shape (2 rows, N columns)
        self.assertEqual(result.shape, (2, 3))

        # Row 1 should be the column names
        # Note: We decode from bytes because of the .astype("S100")
        self.assertEqual(result[0, 0].decode(), "phrog_1")
        self.assertEqual(result[0, 2].decode(), "phrog_99")

        # Row 2 should be the versions
        self.assertEqual(result[1, 0].decode(), "v1.0")
        self.assertEqual(result[1, 1].decode(), "v2.1")
        self.assertEqual(result[1, 2].decode(), "unknown")

    def test_build_info_array_with_source_map(self):
        """Test that source_map correctly overrides version mapping for specific columns."""
        # This simulates your serotype columns
        cols = ["k_type_dim1", "phrog_1"]
        source_map = {"k_type_dim1": "best_match"}

        result = MetadataManager.build_info_array(cols, self.meta_df, source_map=source_map)

        # k_type_dim1 should have 'best_match' as its version/info
        self.assertEqual(result[1, 0].decode(), "best_match")
        # phrog_1 should still map to its version from the meta_df
        self.assertEqual(result[1, 1].decode(), "v1.0")

    def test_empty_metadata_handling(self):
        """Ensure it handles an empty metadata dataframe gracefully."""
        empty_meta = pd.DataFrame(columns=["hmm_id", "clustering_version"])
        result = MetadataManager.build_info_array(self.cols, empty_meta)

        self.assertEqual(result[1, 0].decode(), "unknown")
        self.assertEqual(result[1, 1].decode(), "unknown")

    def test_dtype_consistency(self):
        """Verify the output is specifically a fixed-length byte string array."""
        result = MetadataManager.build_info_array(self.cols, self.meta_df)
        # S100 in numpy is represented by the kind 'S' (zero-terminated bytes)
        self.assertEqual(result.dtype.kind, "S")
        self.assertEqual(result.dtype.itemsize, 100)


if __name__ == "__main__":
    unittest.main()
