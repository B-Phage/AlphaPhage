from pathlib import Path

import duckdb
import h5py
import numpy as np
import pandas as pd
import pytest
from data_contracts.h5_schema import H5DataKeys
from data_contracts.interaction_db import InteractionTableContract
from representation.build_model_data import build_model_data


@pytest.fixture
def setup_test_db(tmp_path):
    """Creates a DuckDB with the full schema required by FeatureExtractor and MetadataManager."""
    db_path = tmp_path / "features.db"
    con = duckdb.connect(str(db_path))

    # FeatureExtractor and MetadataManager expect organism_id, hmm_id, token_id, and clustering_version
    # Based on VARIANT_LAKE_SCHEMA
    for table in ["phage_feats", "host_feats"]:
        con.execute(
            f"""
            CREATE TABLE {table} (
                organism_id VARCHAR,
                hmm_id VARCHAR,
                token_id INT,
                clustering_version VARCHAR
            )
        """
        )

    # Insert sample data including the clustering_version
    con.execute(
        """
        INSERT INTO phage_feats VALUES
        ('p_ready', 'hmm_1', 10, 'v1.0'),
        ('p_ready', 'hmm_2', 20, 'v1.0'),
        ('p_unused', 'hmm_1', 99, 'v1.0')
    """
    )

    con.execute(
        """
        INSERT INTO host_feats VALUES
        ('h_ready', 'hmm_1', 100, 'v1.0'),
        ('h_unused', 'hmm_1', 500, 'v1.0')
    """
    )

    # Serotypes usually don't have clustering_version in these pipelines,
    # but check your extractor if it fails here next!
    con.execute("CREATE TABLE serotypes (organism_id VARCHAR, k_type_dim1 INT, k_type_dim2 INT)")
    con.execute("INSERT INTO serotypes VALUES ('h_ready', 1, 1), ('h_unused', 2, 2)")

    con.close()
    return str(db_path)


@pytest.fixture
def interaction_tsv(tmp_path):
    """Creates a TSV with a mix of valid and invalid references."""
    path = tmp_path / "interactions.tsv"
    # Row 1: Valid (p_ready, h_ready)
    # Row 2: Invalid (p_missing, h_ready) -> Should be dropped
    data = {
        "h_ready": {"p_ready": 1, "p_missing": 0},
    }
    df = pd.DataFrame(data).T
    df.to_csv(path, sep="\t")
    return str(path)


def test_build_model_data_full_flow(setup_test_db, interaction_tsv, tmp_path, caplog):
    out_dir = tmp_path / "output"
    base_name = "prod_run"
    keys = H5DataKeys()

    # EXECUTE
    build_model_data(
        interaction_tsv=interaction_tsv,
        db_path=setup_test_db,
        output_dir=str(out_dir),
        base_filename=base_name,
        phage_table="phage_feats",
        host_table="host_feats",
        serotype_table="serotypes",
    )

    # 1. VERIFY LOGGING (Edge Case: Missing Organism)
    assert "p_missing" in caplog.text
    assert "1 phages in interactions missing features" in caplog.text
    assert "Trimmed unused" in caplog.text

    # 2. VERIFY HDF5 STRUCTURE & DATA
    x_file = out_dir / f"x_{base_name}.h5"
    cl_file = out_dir / f"cl_{base_name}.h5"

    assert x_file.exists()
    assert cl_file.exists()

    with h5py.File(x_file, "r") as f_x:
        # Check Trimming: Only 'p_ready' and 'h_ready' should be in the vectors
        # 'p_unused' and 'h_unused' should have been stripped out.
        assert f_x[keys.PHAGE_VEC].shape[0] == 1
        assert f_x[keys.HOST_VEC].shape[0] == 1

        # Verify specific feature values to ensure alignment didn't scramble data
        # host_vec = [k_type_dim1, k_type_dim2, hmm_1]
        host_data = f_x[keys.HOST_VEC][0]
        assert np.array_equal(host_data, [1, 1, 100])

        # Verify Pair Indices
        # Since we only have 1 phage and 1 host left, index should be [0, 0]
        assert np.array_equal(f_x[keys.PAIR_INDICES][0], [0, 0])

    with h5py.File(cl_file, "r") as f_cl:
        # Check Label Contract
        assert f_cl[keys.LABELS].shape[0] == 1
        assert f_cl[keys.LABELS][0] == 1  # The value for (h_ready, p_ready)


def test_build_model_data_inference_mode(setup_test_db, tmp_path):
    """Verifies that labels with -1 are preserved when is_inference=True."""
    # 1. Setup Inference TSV with -1 labels
    path = tmp_path / "inference_interactions.tsv"
    # We want to predict p_ready vs h_ready
    data = {"h_ready": {"p_ready": -1}}
    pd.DataFrame(data).T.to_csv(path, sep="\t")

    out_dir = tmp_path / "inference_output"
    base_name = "inf_run"
    keys = H5DataKeys()

    # 2. EXECUTE with is_inference=True
    build_model_data(
        interaction_tsv=str(path),
        db_path=setup_test_db,
        output_dir=str(out_dir),
        base_filename=base_name,
        phage_table="phage_feats",
        host_table="host_feats",
        serotype_table="serotypes",
        is_inference=True,  # CRITICAL FLAG
    )

    # 3. VERIFY
    cl_file = out_dir / f"cl_{base_name}.h5"
    with h5py.File(cl_file, "r") as f_cl:
        # If the filter was active, this would be empty.
        # Since it's inference, we expect 1 row with value -1.
        assert f_cl[keys.LABELS].shape[0] == 1
        assert f_cl[keys.LABELS][0] == -1

    # Verify indices still map correctly
    x_file = out_dir / f"x_{base_name}.h5"
    with h5py.File(x_file, "r") as f_x:
        assert np.array_equal(f_x[keys.PAIR_INDICES][0], [0, 0])
