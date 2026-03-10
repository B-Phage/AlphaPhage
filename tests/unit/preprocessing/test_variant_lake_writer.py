import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
from variants.variant_lake_writer import write_variant_shard


class TestVariantLakeWriter(unittest.TestCase):
    def setUp(self):
        """Set up dummy data and paths."""
        self.out_path = Path("/tmp/test_variant.parquet")

        # Valid dummy data matching the contract requirements
        self.valid_df = pd.DataFrame(
            {
                "organism_id": ["ORG_1", "ORG_1"],
                "hmm_id": ["HMM_A", "HMM_B"],
                "variant_id": ["VAR_001", "VAR_002"],
                "token_id": [10, 20],
                "clustering_version": ["v1", "v1"],
            }
        )

    def test_write_variant_shard_missing_ids_raises_error(self):
        """Verify that missing organism_id or hmm_id triggers a ValueError."""
        # Missing organism_id
        invalid_df = pd.DataFrame({"hmm_id": ["HMM_A"], "token_id": [1]})
        with self.assertRaisesRegex(ValueError, "organism_id"):
            write_variant_shard(invalid_df, self.out_path)

        # Missing hmm_id
        invalid_df_2 = pd.DataFrame({"organism_id": ["ORG_1"], "token_id": [1]})
        with self.assertRaisesRegex(ValueError, "hmm_id"):
            write_variant_shard(invalid_df_2, self.out_path)

    @patch("variants.variant_lake_writer.write_parquet_partition")
    @patch("variants.variant_lake_writer.enforce_schema")
    def test_write_variant_shard_calls_infrastructure(self, mock_enforce, mock_write):
        """Ensure the writer delegates to schema enforcement and storage utilities."""
        # Setup mocks
        mock_enforce.return_value = self.valid_df

        write_variant_shard(self.valid_df, self.out_path)

        # 1. Verify Schema Enforcement was called
        from data_contracts.variant_lake import VARIANT_LAKE_SCHEMA

        mock_enforce.assert_called_once()

        # Access by position: args[0] is df, args[1] is the schema
        args, kwargs = mock_enforce.call_args
        self.assertEqual(args[1], VARIANT_LAKE_SCHEMA)

        # 2. Verify Parquet Write
        from data_contracts.variant_lake import VARIANT_LAKE_PRIMARY_KEYS

        mock_write.assert_called_once_with(
            df=self.valid_df, out_path=self.out_path, key_cols=VARIANT_LAKE_PRIMARY_KEYS
        )

    def test_write_variant_shard_empty_dataframe(self):
        """Check behavior with an empty dataframe (should still pass ID check)."""
        empty_df = pd.DataFrame(columns=["organism_id", "hmm_id", "token_id"])

        # We mock the infrastructure to avoid actual IO
        with patch("variants.variant_lake_writer.write_parquet_partition"):
            with patch("variants.variant_lake_writer.enforce_schema") as mock_enforce:
                mock_enforce.return_value = empty_df
                write_variant_shard(empty_df, self.out_path)
                mock_enforce.assert_called_once()


if __name__ == "__main__":
    unittest.main()
