import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import duckdb
import pandas as pd
from external_db_provisioning.extract_interactions import (
    _parse_list_input,
    extract_interaction_matrix,
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

    @patch("external_db_provisioning.extract_interactions.logger")
    @patch("external_db_provisioning.extract_interactions.DuckDBClient")
    def test_empty_source_warning(self, mock_client, mock_logger):
        """Verify behavior when no data matches the selected sources."""
        # Mock empty dataframe result
        mock_inst = mock_client.return_value.__enter__.return_value
        res_mock = MagicMock()
        res_mock.df.return_value = pd.DataFrame()
        mock_inst.execute.return_value = res_mock

        extract_interaction_matrix(
            self.db_path, str(self.output_tsv), ["NonExistent"], self.table_name
        )

        # Ensure the warning logger was triggered
        mock_logger.info.assert_any_call("⚠️ Warning: No data found for sources: ['NonExistent']")


if __name__ == "__main__":
    unittest.main()
