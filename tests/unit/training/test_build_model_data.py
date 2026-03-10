import sys
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import numpy as np
import pandas as pd
from representation.build_model_data import build_model_data, load_interaction_pairs, main


class TestBuildModelData(unittest.TestCase):
    def setUp(self):
        """Set up dummy data for alignment testing."""
        # Create a dummy interaction matrix (TSV format)
        self.tsv_content = "phage\tPhage_1\tPhage_2\n" "Host_A\t1\t0\n" "Host_B\t0\t-1\n"

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
        # Define exactly what the "wide" matrix looks like before unpivoting
        wide_df = pd.DataFrame(
            [[1, 0], [0, -1]], index=["Host_A", "Host_B"], columns=["Phage_1", "Phage_2"]
        )
        wide_df.index.name = "host"  # Ensure the index has a name for reset_index()

        with patch("pandas.read_csv") as mock_read:
            # Tell the mock to return our pre-constructed wide DataFrame
            mock_read.return_value = wide_df

            df = load_interaction_pairs("dummy.tsv")

            # 1. Check if the labels were filtered correctly (-1 removed)
            # Host_A/Phage_1 (1), Host_A/Phage_2 (0), Host_B/Phage_1 (0) -> 3 rows
            self.assertEqual(len(df), 3)

            # 2. Verify we don't have any -1 labels left
            self.assertNotIn(-1, df["label"].values)

            # 3. Verify column names match our expectations in build_model_data
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
        # Setup mocks
        mock_load.return_value = pd.DataFrame(
            {"host": ["Host_A"], "phage": ["Phage_1"], "label": [1]}
        )

        extractor = mock_extractor_cls.return_value
        extractor.get_wide_phage_features.return_value = self.mock_phage_df
        extractor.get_wide_host_features.return_value = self.mock_host_df

        writer = mock_writer_cls.return_value
        writer.write.return_value = (Path("x.h5"), Path("cl.h5"))

        # Run the function
        build_model_data(
            interaction_tsv="pairs.tsv",
            db_path="feat.db",
            output_dir="out/",
            base_filename="test",
            phage_table="p_tab",
            host_table="h_tab",
            serotype_table="s_tab",
        )

        # 1. Check if alignments were correct
        # The payload sent to writer.write is the best way to verify the join worked
        args, _ = writer.write.call_args
        payload = args[0]

        # Verify merged host vecs: [k1, k2, HMM_X]
        # Host_A features from mock_host_df are [1, 10, 100]
        np.testing.assert_array_equal(payload["host_vecs"], [[1, 10, 100]])
        # Phage_1 features from mock_phage_df are [10]
        np.testing.assert_array_equal(payload["phage_vecs"], [[10]])

    @patch("representation.build_model_data.build_model_data")
    def test_cli_parsing(self, mock_build):
        """Verify CLI flags map to correct function arguments."""
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

        mock_build.assert_called_once_with(
            interaction_tsv="interact.tsv",
            db_path="db.db",
            output_dir="output_dir",
            base_filename="base_name",
            phage_table="p_table",
            host_table="h_table",
            serotype_table="s_table",
        )

    def test_cli_missing_args(self):
        """Verify CLI exits on missing required arguments."""
        test_args = ["script_name", "-i", "only_one_arg"]
        with patch.object(sys, "argv", test_args):
            with self.assertRaises(SystemExit):
                with patch("sys.stderr", StringIO()):  # Suppress help text in console
                    main()


if __name__ == "__main__":
    unittest.main()
