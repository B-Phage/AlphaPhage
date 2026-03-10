import unittest

import duckdb
import numpy as np
import pandas as pd
from storage_io.duckdb_client import DuckDBClient

from packages.representation.feature_extractor import FeatureExtractor


class TestFeatureExtractor(unittest.TestCase):
    def setUp(self):
        """Set up an in-memory DuckDB with sample tables."""
        # Create an in-memory duckdb connection
        self.con = duckdb.connect(database=":memory:")

        # Initialize the client normally
        self.client = DuckDBClient(":memory:")

        # Explicitly assign our in-memory connection to the client's internal connection
        # The 'cast' helps if Pylance is being particularly stubborn
        self.client.con = self.con

        self.extractor = FeatureExtractor(self.client)

        # 1. Create dummy Phage Variants (Long Format)
        self.con.execute(
            """
            CREATE TABLE phage_vars AS
            SELECT * FROM (VALUES
                ('P1', 'HMM_A', 10, 'v1'),
                ('P1', 'HMM_B', 20, 'v1'),
                ('P2', 'HMM_A', 15, 'v1')
            ) AS t(organism_id, hmm_id, token_id, clustering_version)
        """
        )

        # 2. Create dummy Host Variants (Long Format)
        self.con.execute(
            """
            CREATE TABLE host_vars AS
            SELECT * FROM (VALUES
                ('H1', 'HMM_X', 100, 'v2'),
                ('H2', 'HMM_X', 105, 'v2'),
                ('H2', 'HMM_Y', 200, 'v2')
            ) AS t(organism_id, hmm_id, token_id, clustering_version)
        """
        )

        # 3. Create Serotype Table
        self.con.execute(
            """
            CREATE TABLE host_sero AS
            SELECT * FROM (VALUES
                ('H1', 1, 2),
                ('H2', 3, 4),
                ('H3', 0, 0)
            ) AS t(organism_id, k_type_dim1, k_type_dim2)
        """
        )

    def tearDown(self):
        self.client.close()

    def test_get_wide_phage_features(self):
        """Verify the PIVOT logic for Phage features."""
        df = self.extractor.get_wide_phage_features("phage_vars")

        # P1 should have both HMMs, P2 should have HMM_B as 0 (due to fillna)
        self.assertEqual(len(df), 2)
        self.assertIn("HMM_A", df.columns)
        self.assertIn("HMM_B", df.columns)

        p1 = df[df["organism_id"] == "P1"].iloc[0]
        self.assertEqual(p1["HMM_A"], 10)
        self.assertEqual(p1["HMM_B"], 20)

        p2 = df[df["organism_id"] == "P2"].iloc[0]
        self.assertEqual(p2["HMM_A"], 15)
        self.assertEqual(p2["HMM_B"], 0)  # Missing value filled with 0

    def test_get_wide_host_features(self):
        """Verify the CTE pivot and LEFT JOIN with serotypes."""
        df = self.extractor.get_wide_host_features("host_vars", "host_sero")

        # H3 exists in serotypes but has NO variants in host_vars
        self.assertEqual(len(df), 3)
        self.assertIn("k_type_dim1", df.columns)
        self.assertIn("HMM_X", df.columns)

        # Verify H2 (Full data)
        h2 = df[df["organism_id"] == "H2"].iloc[0]
        self.assertEqual(h2["k_type_dim1"], 3)
        self.assertEqual(h2["HMM_Y"], 200)

        # Verify H3 (Serotypes present, variants missing/padded)
        h3 = df[df["organism_id"] == "H3"].iloc[0]
        self.assertEqual(h3["k_type_dim1"], 0)
        self.assertEqual(h3["HMM_X"], 0)

    def test_get_metadata(self):
        """Verify extraction of distinct HMM-version pairs."""
        df = self.extractor.get_metadata("phage_vars")

        # Should contain HMM_A/v1 and HMM_B/v1
        self.assertEqual(len(df), 2)
        self.assertTrue(((df["hmm_id"] == "HMM_A") & (df["clustering_version"] == "v1")).any())


if __name__ == "__main__":
    unittest.main()
