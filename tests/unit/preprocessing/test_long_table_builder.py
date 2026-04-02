import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import duckdb
import pandas as pd
from variants.long_table_builder import build_long_table


class TestLongTableBuilder(unittest.TestCase):
    def setUp(self):
        """Set up temporary directories and dummy Parquet data."""
        self.test_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.test_dir.name)

        # 1. Create a dummy Parquet directory and file
        self.parquet_dir = self.temp_path / "shards"
        self.parquet_dir.mkdir()

        self.dummy_data = pd.DataFrame(
            {
                "organism_id": ["ORG_A", "ORG_B", "ORG_C"],
                "hmm_id": ["HMM_1", "HMM_1", "HMM_2"],
                "variant_id": ["var_100", "var_200", "var_999"],  # 999 is not in vocab
                "clustering_version": ["v1", "v1", "v1"],
            }
        )
        self.dummy_data.to_parquet(self.parquet_dir / "shard_1.parquet")

        self.db_path = str(self.temp_path / "test.db")
        self.table_name = "test_variants"
        self.vocab_section = "phage_mapping"
        self.manifest_path = str(self.temp_path / "manifest.json")

        # 2. Mock Vocabulary
        self.mock_vocab = {"phage_mapping": {"var_100": 1, "var_200": 2}}

    def tearDown(self):
        self.test_dir.cleanup()

    @patch("variants.long_table_builder.get_vocab")
    def test_build_long_table_success(self, mock_get_vocab):
        """Verify aggregation, token mapping join, and manifest generation."""
        mock_get_vocab.return_value = self.mock_vocab

        build_long_table(
            parquet_dir=str(self.parquet_dir),
            db_path=self.db_path,
            table_name=self.table_name,
            vocab_section=self.vocab_section,
            manifest_path=self.manifest_path,
        )

        # 3. Verify Database Content
        con = duckdb.connect(self.db_path)
        df = con.execute(f"SELECT * FROM {self.table_name} ORDER BY organism_id").df()
        con.close()

        # Check mapped tokens
        self.assertEqual(len(df), 3)
        self.assertEqual(df[df["organism_id"] == "ORG_A"]["token_id"].iloc[0], 1)
        self.assertEqual(df[df["organism_id"] == "ORG_B"]["token_id"].iloc[0], 2)

        # Check LEFT JOIN behavior (missing vocab entry should be NaN/NULL)
        # Note: token_id is float in pandas if it contains NaNs
        self.assertTrue(pd.isna(df[df["organism_id"] == "ORG_C"]["token_id"].iloc[0]))

    @patch("variants.long_table_builder.get_vocab")
    def test_manifest_output(self, mock_get_vocab):
        """Verify the JSON manifest is created with correct keys."""
        mock_get_vocab.return_value = self.mock_vocab

        build_long_table(
            parquet_dir=str(self.parquet_dir),
            db_path=self.db_path,
            table_name=self.table_name,
            vocab_section=self.vocab_section,
            manifest_path=self.manifest_path,
        )

        manifest_file = Path(self.manifest_path)
        self.assertTrue(manifest_file.exists())

        with open(manifest_file, "r") as f:
            data = json.load(f)
            self.assertEqual(data["table"], self.table_name)
            self.assertEqual(data["source"], str(self.parquet_dir))
            self.assertIn("fingerprint", data)

    @patch("variants.long_table_builder.get_vocab")
    def test_type_casting_join(self, mock_get_vocab):
        """Ensure variant_ids as integers in vocab still join with string IDs in Parquet."""
        # Vocab has integer keys
        numeric_vocab = {"phage_mapping": {100: 1}}
        mock_get_vocab.return_value = numeric_vocab

        # Parquet has string ID '100'
        pd.DataFrame(
            {
                "organism_id": ["ORG_X"],
                "hmm_id": ["HMM_1"],
                "variant_id": ["100"],
                "clustering_version": ["v1"],
            }
        ).to_parquet(self.parquet_dir / "shard_2.parquet")

        # Cleanup table if exists
        if Path(self.db_path).exists():
            Path(self.db_path).unlink()

        build_long_table(
            parquet_dir=str(self.parquet_dir),
            db_path=self.db_path,
            table_name=self.table_name,
            vocab_section=self.vocab_section,
        )

        con = duckdb.connect(self.db_path)
        token = con.execute(
            f"SELECT token_id FROM {self.table_name} WHERE organism_id='ORG_X'"
        ).fetchone()[0]
        con.close()

        # The 'CAST(... AS VARCHAR)' in the SQL should bridge '100' and 100
        self.assertEqual(token, 1)

    @patch("variants.long_table_builder.build_long_table")
    def test_cli_parsing(self, mock_build):
        """Verify that CLI arguments are correctly parsed and passed to the builder."""
        # Simulate command line arguments
        test_args = [
            "prog",  # placeholder for script name
            "-i",
            "/path/to/parquet",
            "-d",
            "features.db",
            "-t",
            "phage_vars",
            "-s",
            "phage_mapping",
            "-mp",
            "manifest.json",
        ]

        # Patch sys.argv to simulate running from the terminal
        with patch.object(sys, "argv", test_args):
            from variants.long_table_builder import cli

            cli()

        # Verify build_long_table was called with the correct mapped arguments
        mock_build.assert_called_once_with(
            parquet_dir="/path/to/parquet",
            db_path="features.db",
            table_name="phage_vars",
            vocab_section="phage_mapping",
            config_path=None,  # Not provided in test_args
            manifest_path="manifest.json",
        )

    @patch("variants.long_table_builder.build_long_table")
    def test_cli_optional_config(self, mock_build):
        """Verify that the optional config_path (-c) is correctly parsed."""
        test_args = [
            "prog",
            "-i",
            "in",
            "-d",
            "out.db",
            "-t",
            "table",
            "-s",
            "section",
            "-c",
            "custom_vocab.yaml",
        ]

        with patch.object(sys, "argv", test_args):
            from variants.long_table_builder import cli

            cli()

        # Check that config_path was picked up
        _, kwargs = mock_build.call_args
        self.assertEqual(kwargs["config_path"], "custom_vocab.yaml")


if __name__ == "__main__":
    unittest.main()
