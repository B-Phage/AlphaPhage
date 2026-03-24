import sys
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import ANY, MagicMock, patch

import numpy as np
import pandas as pd
from representation.build_model_data import build_model_data, load_interaction_pairs, main


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

    @patch("representation.build_model_data.pd.read_csv")
    @patch("representation.build_model_data.Path.exists")
    def test_align_all_features_tsv_layouts(self, mock_exists, mock_read):
        """Verify that both vertical and horizontal TSVs result in correct feature counts."""
        from representation.build_model_data import _align_all_features

        mock_exists.return_value = True

        # Scenario A: Vertical TSV (Many rows, 1 column)
        # This was the layout causing the bug
        vertical_df = pd.DataFrame({"0": ["HMM_1", "HMM_2", "HMM_3"]})
        mock_read.return_value = vertical_df

        # We only need to test the phage side for simplicity
        _, p_cols, _, _ = _align_all_features(
            self.mock_phage_df,
            self.mock_host_df,
            phage_ref_path="phage.tsv",
            host_ref_path=None,
            m_token=0,
        )
        self.assertEqual(len(p_cols), 3, "Vertical TSV failed to find all 3 features")
        self.assertListEqual(p_cols, ["HMM_1", "HMM_2", "HMM_3"])

        # Scenario B: Horizontal TSV (1 row, many columns)
        horizontal_df = pd.DataFrame([["HMM_1", "HMM_2", "HMM_3"]])
        mock_read.return_value = horizontal_df

        _, p_cols, _, _ = _align_all_features(
            self.mock_phage_df,
            self.mock_host_df,
            phage_ref_path="phage.tsv",
            host_ref_path=None,
            m_token=0,
        )
        self.assertEqual(len(p_cols), 3, "Horizontal TSV failed to find all 3 features")

    @patch("representation.build_model_data.pd.read_csv")
    @patch("representation.build_model_data.Path.exists")
    def test_align_forces_reference_length(self, mock_exists, mock_read):
        """Verify that the output width matches the reference, even if data is missing."""
        from representation.build_model_data import _align_all_features

        mock_exists.return_value = True

        # Reference expects 5 features
        ref_df = pd.DataFrame([["F1", "F2", "F3", "F4", "F5"]])
        mock_read.return_value = ref_df

        # Input data only has 1 feature ("token_1" which isn't in the ref)
        # This simulates the "silent trim" where everything becomes a missing_token
        p_df, p_cols, _, _ = _align_all_features(
            self.mock_phage_df,
            self.mock_host_df,
            phage_ref_path="phage.tsv",
            host_ref_path=None,
            m_token=99,
        )

        # Width must be exactly 5
        self.assertEqual(len(p_cols), 5)
        self.assertEqual(p_df.shape[1], 6)  # 5 features + 1 organism_id

        # Check that missing data was padded with the m_token (99)
        # Taking the first sample, first feature
        self.assertEqual(p_df.iloc[0]["F1"], 99)

    @patch("representation.build_model_data.pd.read_csv")
    @patch("representation.build_model_data.Path.exists")
    def test_host_alignment_preserves_serotypes(self, mock_exists, mock_read):
        """Ensure k_type columns are preserved even if HMM alignment shrinks."""
        from representation.build_model_data import _align_all_features

        mock_exists.return_value = True

        # Ref only has 1 HMM
        mock_read.return_value = pd.DataFrame([["HMM_X"]])

        _, _, h_df, h_cols = _align_all_features(
            self.mock_phage_df,
            self.mock_host_df,
            phage_ref_path=None,
            host_ref_path="host.tsv",
            m_token=0,
        )

        # Final columns should be: [k_type_dim1, k_type_dim2, HMM_X]
        self.assertEqual(len(h_cols), 3)
        self.assertEqual(h_cols[0], "k_type_dim1")
        self.assertEqual(h_cols[1], "k_type_dim2")
        self.assertEqual(h_cols[2], "HMM_X")

        # Verify the data merged correctly
        self.assertEqual(h_df.iloc[0]["k_type_dim1"], 1)
        self.assertEqual(h_df.iloc[0]["HMM_X"], 100)


if __name__ == "__main__":
    unittest.main()
