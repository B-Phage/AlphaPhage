import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import duckdb
import pandas as pd
from storage_io.duckdb_client import DuckDBClient

from packages.representation.typing_aggregator import aggregate_serotypes, main


class TestTypingAggregator(unittest.TestCase):
    def setUp(self):
        """Set up temporary directory and files."""
        self.test_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.test_dir.name)

        # 1. Create dummy Kaptive JSON files
        self.json_dir = self.temp_path / "kaptive_results"
        self.json_dir.mkdir()

        # Sample 1: Explicit sample name
        with open(self.json_dir / "sample1.json", "w") as f:
            json.dump({"sample_name": "ORG_001", "best_match": "KL10"}, f)

        # Sample 2: Fallback to filename (no sample_name in JSON)
        with open(self.json_dir / "ORG_002.json", "w") as f:
            json.dump({"best_match": "KL11"}, f)

        # Sample 3: Unknown serotype (not in vocab)
        with open(self.json_dir / "ORG_003.json", "w") as f:
            json.dump({"best_match": "KL_UNKNOWN"}, f)

        self.db_path = str(self.temp_path / "test_features.db")
        self.table_name = "test_serotypes"
        self.manifest_path = str(self.temp_path / "manifest.json")

        # 2. Mock Vocab Structure
        self.mock_vocab = MagicMock()
        self.mock_vocab.vocab_best_match = {
            "KL10": [10, 100],
            "KL11": [11, 111],
            "-": [0, 0],  # The fallback
        }
        # Ensure 'in' operator works for the KeyError check in script
        self.mock_vocab.__contains__.side_effect = lambda x: x == "vocab_best_match"

    def tearDown(self):
        self.test_dir.cleanup()

    @patch("packages.representation.typing_aggregator.get_vocab")
    def test_aggregate_serotypes_success(self, mock_get_vocab):
        """Test successful aggregation and token mapping."""
        mock_get_vocab.return_value = self.mock_vocab

        aggregate_serotypes(
            json_dir=str(self.json_dir),
            db_path=self.db_path,
            table_name=self.table_name,
            manifest_path=self.manifest_path,
        )

        # Verify Database Content
        with DuckDBClient(self.db_path, read_only=True) as client:
            df = client.execute(f"SELECT * FROM {self.table_name} ORDER BY organism_id").df()

            # Check ORG_001 (KL10)
            row1 = df[df["organism_id"] == "ORG_001"].iloc[0]
            self.assertEqual(row1["k_type_dim1"], 10)
            self.assertEqual(row1["k_type_dim2"], 100)

            # Check ORG_002 (Filename fallback)
            self.assertIn("ORG_002", df["organism_id"].values)
            row2 = df[df["organism_id"] == "ORG_002"].iloc[0]
            self.assertEqual(row2["k_type_dim1"], 11)

            # Check ORG_003 (Unknown fallback to '-')
            row3 = df[df["organism_id"] == "ORG_003"].iloc[0]
            self.assertEqual(row3["k_type_dim1"], 0)
            self.assertEqual(row3["k_type_dim2"], 0)

    @patch("packages.representation.typing_aggregator.get_vocab")
    def test_missing_vocab_section_raises_error(self, mock_get_vocab):
        """Ensure script fails fast if vocab section is missing."""
        mock_get_vocab.return_value = {}  # Empty dict

        with self.assertRaises(KeyError):
            aggregate_serotypes(
                str(self.json_dir), self.db_path, self.table_name, self.manifest_path
            )

    @patch("packages.representation.typing_aggregator.get_vocab")
    def test_manifest_creation(self, mock_get_vocab):
        """Verify the DVC manifest is written with the correct structure."""
        mock_get_vocab.return_value = self.mock_vocab

        aggregate_serotypes(str(self.json_dir), self.db_path, self.table_name, self.manifest_path)

        manifest_file = Path(self.manifest_path)
        self.assertTrue(manifest_file.exists())

        with open(manifest_file, "r") as f:
            data = json.load(f)
            self.assertEqual(data["table"], self.table_name)
            self.assertEqual(data["type"], "serotype_token_aggregation")
            self.assertIn("fingerprint", data)

    @patch("packages.representation.typing_aggregator.aggregate_serotypes")
    def test_main_argument_parsing(self, mock_aggregate):
        """Verify that CLI flags are correctly mapped to function arguments."""
        # 1. Simulate the command line arguments
        test_args = [
            "prog_name",
            "-i",
            "/data/kaptive_jsons",
            "-d",
            "features.db",
            "-t",
            "bacteria_serotypes_v1",
            "-m",
            "manifest.json",
            "-v",
            "custom_vocab.yaml",
        ]

        # 2. Patch sys.argv and call main
        with patch.object(sys, "argv", test_args):
            main()

        # 3. Assert aggregate_serotypes was called with the correct parameters
        mock_aggregate.assert_called_once_with(
            json_dir="/data/kaptive_jsons",
            db_path="features.db",
            table_name="bacteria_serotypes_v1",
            manifest_path="manifest.json",
            vocab_path="custom_vocab.yaml",
        )

    @patch("packages.representation.typing_aggregator.aggregate_serotypes")
    def test_main_optional_vocab_is_none(self, mock_aggregate):
        """Verify that vocab_path is None if the -v flag is omitted."""
        test_args = [
            "prog_name",
            "-i",
            "input_dir",
            "-d",
            "db.db",
            "-t",
            "table",
            "-m",
            "manifest.json",
            # -v is intentionally missing
        ]

        with patch.object(sys, "argv", test_args):
            main()

        # Check that the last argument (vocab_path) was passed as None
        _, kwargs = mock_aggregate.call_args
        self.assertIsNone(kwargs["vocab_path"])

    def test_main_missing_required_args(self):
        """Verify that argparse exits if required arguments are missing."""
        # Missing -i and -d
        test_args = ["prog_name", "-t", "table", "-m", "manifest.json"]

        with patch.object(sys, "argv", test_args):
            # argparse calls sys.exit(2) on error
            with self.assertRaises(SystemExit) as cm:
                # We suppress stderr to keep test output clean
                with patch("sys.stderr", MagicMock()):
                    main()

            self.assertEqual(cm.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
