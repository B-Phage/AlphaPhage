import ast
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import duckdb
import pandas as pd
from external_db_provisioning.extract_interactions import (
    _parse_list_input,
    extract_interaction_matrix,
    main,
)


class TestExtractInteractions(unittest.TestCase):
    def setUp(self):
        """Set up an in-memory database with interaction data."""
        self.test_dir = tempfile.TemporaryDirectory()
        self.output_tsv = Path(self.test_dir.name) / "matrix.tsv"
        self.db_path = ":memory:"
        self.table_name = "test_interactions"

        # Initialize DuckDB with dummy data
        # Source_A: 1 (Positive), Source_B: 0 (Negative), Source_C: Untrusted
        self.con = duckdb.connect(self.db_path)
        self.con.execute(
            f"""
            CREATE TABLE {self.table_name} AS
            SELECT * FROM (VALUES
                ('Bact_1', 'Phage_X', 1, 'Source_A'),
                ('Bact_1', 'Phage_Y', 0, 'Source_B'),
                ('Bact_2', 'Phage_X', 1, 'Source_A'),
                ('Bact_2', 'Phage_Z', 1, 'Source_C')
            ) AS t(host, phage, interaction, source)
        """
        )

    def tearDown(self):
        self.con.close()
        self.test_dir.cleanup()

    @patch("external_db_provisioning.extract_interactions.DuckDBClient")
    def test_extract_interaction_matrix_pivot(self, mock_client):
        """Verify filtering for trusted sources and pivoting to wide format."""
        # Only trust Source_A and Source_B
        selected_sources = ["Source_A", "Source_B"]

        # Setup the mock context manager behavior for DuckDBClient
        mock_inst = mock_client.return_value.__enter__.return_value
        # This allows validate_schema to pass
        mock_inst.list_columns.return_value = ["host", "phage", "interaction", "source"]

        # Redirect the mock's execute calls to our live in-memory connection
        # We wrap the result in a MagicMock that has a .df() method returning the expected pandas DF
        def mock_execute(query, params=None):
            rel = self.con.execute(query)
            # Create a mock result object that mimics the DuckDB relation/connection API
            res_mock = MagicMock()
            res_mock.df.return_value = rel.df()
            return res_mock

        mock_inst.execute.side_effect = mock_execute

        extract_interaction_matrix(
            source_db_path=self.db_path,
            output_tsv=str(self.output_tsv),
            selected_sources=selected_sources,
            table_name=self.table_name,
        )

        # Load and verify the resulting TSV
        self.assertTrue(self.output_tsv.exists())
        df = pd.read_csv(self.output_tsv, sep="\t", index_col=0)

        # 1. Check Dimensions
        # Bact_2 only has Phage_X data from Source_A. Phage_Y column should be created and filled with -1.
        self.assertEqual(df.shape, (2, 2))
        self.assertIn("Phage_X", df.columns)
        self.assertIn("Phage_Y", df.columns)

        # 2. Verify values (1=Pos, 0=Neg, -1=Untested/Missing)
        self.assertEqual(df.loc["Bact_1", "Phage_X"], 1)
        self.assertEqual(df.loc["Bact_1", "Phage_Y"], 0)
        self.assertEqual(df.loc["Bact_2", "Phage_X"], 1)
        self.assertEqual(df.loc["Bact_2", "Phage_Y"], -1)

    def test_parse_list_input_variations(self):
        """Test the robust parsing of various string/list input formats."""
        # Already a list
        self.assertEqual(_parse_list_input(["A", "B"]), ["A", "B"])

        # Stringified Python list
        self.assertEqual(_parse_list_input("['Source_1', 'Source_2']"), ["Source_1", "Source_2"])

        # Space-separated string (fallback)
        self.assertEqual(_parse_list_input("Source_A Source_B"), ["Source_A", "Source_B"])

        # Empty input handling
        self.assertEqual(_parse_list_input(None), [])
        self.assertEqual(_parse_list_input(""), [])

        # This now passes because the function checks for str type
        self.assertEqual(_parse_list_input(123), [123])

        # Test a string that literal_evals to an int
        # (This triggers the final 'return [input_val]')
        self.assertEqual(_parse_list_input("456"), ["456"])

        # Test a single quoted string
        self.assertEqual(_parse_list_input("'Source_Z'"), ["'Source_Z'"])

    @patch("external_db_provisioning.extract_interactions.logger")
    @patch("external_db_provisioning.extract_interactions.DuckDBClient")
    def test_empty_source_warning(self, mock_client, mock_logger):
        """Verify behavior when no data matches the selected sources."""
        # Mock empty dataframe result
        mock_inst = mock_client.return_value.__enter__.return_value
        mock_inst.list_columns.return_value = ["host", "phage", "interaction", "source"]
        res_mock = MagicMock()
        res_mock.df.return_value = pd.DataFrame()
        mock_inst.execute.return_value = res_mock

        extract_interaction_matrix(
            self.db_path, str(self.output_tsv), ["NonExistent"], self.table_name
        )

        # Ensure the warning logger was triggered
        mock_logger.info.assert_any_call("⚠️ Warning: No data found for sources: ['NonExistent']")


class TestExtractInteractionsMain(unittest.TestCase):

    @patch("external_db_provisioning.extract_interactions.get_config")
    @patch("external_db_provisioning.extract_interactions.extract_interaction_matrix")
    def test_main_orchestration(self, mock_extract, mock_get_config):
        """Verify main() loads config and passes correct values to extractor."""

        # 1. Setup Mock Config Object
        mock_cfg = MagicMock()
        mock_cfg.source_db = "path/to/source.db"
        mock_cfg.output_matrix = "path/to/output.tsv"
        mock_cfg.trusted_sources = ["Lab_A", "Lab_B"]
        mock_cfg.table_name = "custom_interaction_table"

        mock_get_config.return_value = mock_cfg

        # 2. Simulate CLI call: python extract_interactions.py -c custom_params.yaml
        test_args = ["prog", "-c", "custom_params.yaml"]

        with patch.object(sys, "argv", test_args):
            main()

        # 3. Assertions
        # Check if config was loaded with the path from CLI
        mock_get_config.assert_called_once_with(
            override_path="custom_params.yaml", params_set="data_intake"
        )

        # Check if the extraction function received the values from the mock config
        mock_extract.assert_called_once_with(
            source_db_path="path/to/source.db",
            output_tsv="path/to/output.tsv",
            selected_sources=["Lab_A", "Lab_B"],
            table_name="custom_interaction_table",
        )

    @patch("external_db_provisioning.extract_interactions.get_config")
    @patch("external_db_provisioning.extract_interactions.extract_interaction_matrix")
    def test_main_optional_table_name(self, mock_extract, mock_get_config):
        """Ensure it handles a missing table_name in the config (AttributeError fallback)."""

        # Setup mock config WITHOUT table_name attribute
        mock_cfg = MagicMock()
        mock_cfg.source_db = "db.db"
        mock_cfg.output_matrix = "out.tsv"
        mock_cfg.trusted_sources = ["S1"]
        # Explicitly delete table_name if MagicMock created it automatically
        if hasattr(mock_cfg, "table_name"):
            del mock_cfg.table_name

        mock_get_config.return_value = mock_cfg

        with patch.object(sys, "argv", ["prog"]):
            main()

        # Verify table_name was passed as None (so the function uses Contract default)
        _, kwargs = mock_extract.call_args
        self.assertIsNone(kwargs["table_name"])

    @patch("external_db_provisioning.extract_interactions.DuckDBClient")
    def test_validate_schema_missing_column_raises_error(self, mock_client):
        """Verify that validate_schema raises KeyError when a required column is missing."""
        from external_db_provisioning.extract_interactions import (
            InteractionTableContract,
            validate_schema,
        )

        mock_inst = mock_client.return_value.__enter__.return_value
        contract = InteractionTableContract()

        # 1. Setup mock to return a list MISSING 'source'
        mock_inst.list_columns.return_value = ["host", "phage", "interaction"]

        # 2. Assert that calling validate_schema raises the specific KeyError
        with self.assertRaises(KeyError) as cm:
            validate_schema(mock_inst, "some_table", contract)

        # 3. Optional: Verify the error message contains the helpful details
        self.assertIn("missing required column: 'source'", str(cm.exception))
        self.assertIn("Found: ['host', 'phage', 'interaction']", str(cm.exception))

    def test_validate_schema_success(self):
        """Verify that validate_schema passes silently when all columns are present."""
        from external_db_provisioning.extract_interactions import (
            InteractionTableContract,
            validate_schema,
        )

        # Create a mock client that returns the full set of columns
        mock_client = MagicMock()
        mock_client.list_columns.return_value = ["host", "phage", "interaction", "source"]
        contract = InteractionTableContract()

        # This should not raise any exception
        try:
            validate_schema(mock_client, "good_table", contract)
        except KeyError:
            self.fail("validate_schema raised KeyError unexpectedly!")


if __name__ == "__main__":
    unittest.main()
