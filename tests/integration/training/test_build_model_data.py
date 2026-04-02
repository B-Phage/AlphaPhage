import sys
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import ANY, MagicMock, patch

import numpy as np
import pandas as pd
from representation.build_model_data import (
    build_model_data,
    load_interaction_pairs,
    main,
)


class TestBuildModelData(unittest.TestCase):
    def setUp(self):
        """Set up dummy data for alignment testing."""
        # Mock DataFrames representing wide features
        self.mock_phage_df = pd.DataFrame(
            {"organism_id": ["Phage_1", "Phage_2"], "token_1": [10, 11]}
        )
        self.mock_host_df = pd.DataFrame(
            {
                "organism_id": ["Host_A", "Host_B"],
                "k_type_dim1": [1, 2],
                "k_type_dim2": [10, 20],
                "HMM_X": [100, 101],
            }
        )

    def test_load_interaction_pairs(self):
        """Verify unpivoting logic and filtering of -1 labels."""
        # Create a wide matrix as if read from TSV
        wide_df = pd.DataFrame(
            [[1, 0], [0, -1]], index=["Host_A", "Host_B"], columns=["Phage_1", "Phage_2"]
        )
        wide_df.index.name = "host"

        with patch("pandas.read_csv") as mock_read:
            mock_read.return_value = wide_df

            # Default: Training mode (drops -1)
            df = load_interaction_pairs("dummy.tsv", include_unknown=False)

            # Host_A/Phage_1 (1), Host_A/Phage_2 (0), Host_B/Phage_1 (0) -> 3 rows
            self.assertEqual(len(df), 3)
            self.assertNotIn(-1, df["label"].values)
            self.assertListEqual(list(df.columns), ["host", "phage", "label"])

    @patch("representation.build_model_data.DuckDBClient")
    @patch("representation.build_model_data.FeatureExtractor")
    @patch("representation.build_model_data.MetadataManager")
    @patch("representation.build_model_data.H5TrainingWriter")
    @patch("representation.build_model_data.load_interaction_pairs")
    def test_orchestration_flow(
        self, mock_load, mock_writer_cls, mock_meta, mock_extractor_cls, mock_db
    ):
        """Verify the full orchestration: Extract -> Align -> Meta -> Write."""
        # 1. Setup mocks
        # Only Host_A and Phage_1 are in the interaction set
        mock_load.return_value = pd.DataFrame(
            {"host": ["Host_A"], "phage": ["Phage_1"], "label": [1]}
        )

        extractor = mock_extractor_cls.return_value
        extractor.get_wide_phage_features.return_value = self.mock_phage_df
        extractor.get_wide_host_features.return_value = self.mock_host_df
        extractor.missing_token = 0

        writer = mock_writer_cls.return_value
        writer.write.return_value = (Path("x.h5"), Path("cl.h5"))

        # 2. Run the function
        build_model_data(
            interaction_tsv="pairs.tsv",
            db_path="feat.db",
            output_dir="out/",
            base_filename="test",
            phage_table="p_tab",
            host_table="h_tab",
            serotype_table="s_tab",
        )

        # 3. Verify writer was called with correct TRIMMED data
        # We check kwargs because our refactored writer uses named arguments
        _, kwargs = writer.write.call_args

        # Host_A features: [k_type_dim1, k_type_dim2, HMM_X] -> [1, 10, 100]
        # It should only have 1 row because Host_B was trimmed
        host_vecs = kwargs["host_vecs"]
        self.assertEqual(host_vecs.shape[0], 1)
        np.testing.assert_array_equal(host_vecs, [[1, 10, 100]])

        # Phage_1 features: [token_1] -> [10]
        # It should only have 1 row because Phage_2 was trimmed
        phage_vecs = kwargs["phage_vecs"]
        self.assertEqual(phage_vecs.shape[0], 1)
        np.testing.assert_array_equal(phage_vecs, [[10]])

        # Pair indices should be [[0, 0]] because both are at index 0 of their trimmed dfs
        np.testing.assert_array_equal(kwargs["pair_indices"], [[0, 0]])

    @patch("representation.build_model_data.build_model_data")
    def test_cli_parsing(self, mock_build):
        """Verify CLI flags map to correct function arguments including defaults."""
        test_args = [
            "script_name",
            "-i",
            "interact.tsv",
            "-d",
            "db.db",
            "-o",
            "output_dir",
            "-f",
            "base_name",
            "--phage_table",
            "p_table",
            "--host_table",
            "h_table",
            "--serotype_table",
            "s_table",
        ]

        with patch.object(sys, "argv", test_args):
            main()

        # Check all arguments, including the ones that default to None/False
        mock_build.assert_called_once_with(
            interaction_tsv="interact.tsv",
            db_path="db.db",
            output_dir="output_dir",
            base_filename="base_name",
            phage_table="p_table",
            host_table="h_table",
            serotype_table="s_table",
            phage_ref_path=None,
            host_ref_path=None,
            vocab_path=None,
            is_inference=False,
        )

    def test_cli_missing_args(self):
        """Verify CLI exits on missing required arguments."""
        test_args = ["script_name", "-i", "only_one_arg"]
        with patch.object(sys, "argv", test_args):
            with self.assertRaises(SystemExit):
                with patch("sys.stderr", StringIO()):
                    main()


if __name__ == "__main__":
    unittest.main()
