import tempfile
import unittest
from pathlib import Path

import duckdb
import h5py
import numpy as np
import pandas as pd
from data_contracts.h5_schema import H5DataKeys
from representation.build_model_data import build_model_data


class TestBuildModelDataIntegration(unittest.TestCase):
    def setUp(self):
        """Set up a real temporary environment with DuckDB and TSV data."""
        self.test_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.test_dir.name)

        self.db_path = str(self.temp_path / "integration_test.db")
        self.interaction_tsv = str(self.temp_path / "interactions.tsv")
        self.output_dir = str(self.temp_path / "output")
        self.base_filename = "run_v1"

        # Table names
        self.phage_table = "phage_vars"
        self.host_table = "host_vars"
        self.serotype_table = "host_serotypes"

        # 1. Create the Interaction TSV
        # Host_1 + Phage_1 = Positive (1)
        # Host_2 + Phage_1 = Negative (0)
        # Host_2 + Phage_2 = Untested (-1) -> Should be ignored
        interact_df = pd.DataFrame(
            [[1, 0], [0, -1]], index=["Host_1", "Host_2"], columns=["Phage_1", "Phage_2"]
        )
        interact_df.index.name = "phage"
        interact_df.to_csv(self.interaction_tsv, sep="\t")

        # 2. Populate DuckDB with real tables
        con = duckdb.connect(self.db_path)

        # Phage Variants (Long Format)
        con.execute(
            f"""
            CREATE TABLE {self.phage_table} AS
            SELECT * FROM (VALUES
                ('Phage_1', 'HMM_P1', 10, 'v1.0'),
                ('Phage_2', 'HMM_P1', 11, 'v1.0')
            ) AS t(organism_id, hmm_id, token_id, clustering_version)
        """
        )

        # Host Variants (Long Format)
        con.execute(
            f"""
            CREATE TABLE {self.host_table} AS
            SELECT * FROM (VALUES
                ('Host_1', 'HMM_H1', 100, 'v2.1'),
                ('Host_2', 'HMM_H1', 101, 'v2.1')
            ) AS t(organism_id, hmm_id, token_id, clustering_version)
        """
        )

        # Host Serotypes (Wide Format)
        con.execute(
            f"""
            CREATE TABLE {self.serotype_table} AS
            SELECT * FROM (VALUES
                ('Host_1', 5, 50),
                ('Host_2', 6, 60)
            ) AS t(organism_id, k_type_dim1, k_type_dim2)
        """
        )
        con.close()

    def tearDown(self):
        self.test_dir.cleanup()

    def test_end_to_end_pipeline(self):
        """Run build_model_data and verify HDF5 contents without mocking."""

        # Run the actual orchestration function
        build_model_data(
            interaction_tsv=self.interaction_tsv,
            db_path=self.db_path,
            output_dir=self.output_dir,
            base_filename=self.base_filename,
            phage_table=self.phage_table,
            host_table=self.host_table,
            serotype_table=self.serotype_table,
        )

        # 1. Verify files exist
        x_path = Path(self.output_dir) / f"x_{self.base_filename}.h5"
        cl_path = Path(self.output_dir) / f"cl_{self.base_filename}.h5"

        self.assertTrue(x_path.exists(), "Feature H5 file was not created.")
        self.assertTrue(cl_path.exists(), "Label H5 file was not created.")

        # 2. Verify HDF5 Content Integrity
        keys = H5DataKeys()
        with h5py.File(x_path, "r") as f:
            # We expected 3 pairs because of the -1 filtering in load_interaction_pairs
            # (Host_1, Phage_1), (Host_1, Phage_2), (Host_2, Phage_1)
            phage_vecs = f[keys.PHAGE_VEC][:]
            host_vecs = f[keys.HOST_VEC][:]

            self.assertEqual(phage_vecs.shape[0], 3)
            self.assertEqual(host_vecs.shape[0], 3)

            # Check Host Vec structure: [k_dim1, k_dim2, HMM_H1]
            # Host_1 should be [5, 50, 100]
            self.assertEqual(host_vecs[0, 0], 5)
            self.assertEqual(host_vecs[0, 1], 50)
            self.assertEqual(host_vecs[0, 2], 100)

            # Check Metadata shapes (Phage has 1 HMM, Host has 1 HMM + 2 Serotype slots = 3)
            self.assertEqual(f[keys.PHAGE_VEC_INFO].shape, (2, 1))
            self.assertEqual(f[keys.HOST_VEC_INFO].shape, (2, 3))

        with h5py.File(cl_path, "r") as f:
            labels = f[keys.PAIR_INT][:]
            self.assertEqual(len(labels), 3)
            # First interaction was Host_1 + Phage_1 = 1
            self.assertEqual(labels[0], 1)


if __name__ == "__main__":
    unittest.main()
