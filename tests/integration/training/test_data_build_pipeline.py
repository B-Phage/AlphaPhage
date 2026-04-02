from pathlib import Path

import duckdb
import h5py
import numpy as np
import pandas as pd
import pytest
from data_contracts.h5_schema import H5DataKeys
from representation.build_model_data import build_model_data


@pytest.fixture
def setup_test_db(tmp_path):
    """
    Creates a DuckDB with the LONG-FORMAT schema.
    Matches VARIANT_LAKE_SCHEMA and PIVOT requirements.
    """
    db_path = tmp_path / "features.db"
    con = duckdb.connect(str(db_path))

    # Feature tables must have hmm_id for PIVOT and clustering_version for Metadata
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

    # p_ready: 2 HMMs | p_unused: 1 HMM (should be trimmed)
    con.execute(
        """
        INSERT INTO phage_feats VALUES
        ('p_ready', 'hmm_1', 10, 'v1'), ('p_ready', 'hmm_2', 20, 'v1'),
        ('p_unused', 'hmm_1', 99, 'v1')
    """
    )

    # h_ready: 1 HMM | h_unused: 1 HMM (should be trimmed)
    con.execute(
        """
        INSERT INTO host_feats VALUES
        ('h_ready', 'hmm_1', 100, 'v1'),
        ('h_unused', 'hmm_1', 500, 'v1')
    """
    )

    # Serotype table (Wide format, usually not pivoted)
    con.execute("CREATE TABLE serotypes (organism_id VARCHAR, k_type_dim1 INT, k_type_dim2 INT)")
    con.execute("INSERT INTO serotypes VALUES ('h_ready', 1, 1), ('h_unused', 2, 2)")

    con.close()
    return str(db_path)


@pytest.fixture
def interaction_tsv(tmp_path):
    """Creates a TSV with a valid pair and a missing-feature edge case."""
    path = tmp_path / "interactions.tsv"
    # Row 1: Valid (p_ready, h_ready)
    # Row 2: Invalid (p_missing, h_ready) -> Should be dropped & logged
    data = {
        "h_ready": {"p_ready": 1, "p_missing": 0},
    }
    df = pd.DataFrame(data).T
    df.index.name = "host"
    df.to_csv(path, sep="\t")
    return str(path)


def test_build_model_data_full_flow(setup_test_db, interaction_tsv, tmp_path, caplog):
    """Tests the full ETL pipeline from DuckDB to HDF5."""
    import logging

    # Ensure we capture INFO level logs as well
    caplog.set_level(logging.INFO)
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

    # 1. VERIFY LOGGING & FILTERING
    # The warning should list the phage missing from the DB
    assert "p_missing" in caplog.text
    assert "1 phages in interactions missing features" in caplog.text
    # Check trimming log (p_unused and h_unused removed)
    assert "Trimmed unused" in caplog.text

    # 2. VERIFY HDF5 OUTPUTS
    x_file = out_dir / f"x_{base_name}.h5"
    cl_file = out_dir / f"cl_{base_name}.h5"

    assert x_file.exists()
    assert cl_file.exists()

    with h5py.File(x_file, "r") as f_x:
        # TRIMMING CHECK: Only 'p_ready' and 'h_ready' should exist
        assert f_x[keys.PHAGE_VEC].shape[0] == 1
        assert f_x[keys.HOST_VEC].shape[0] == 1

        # ALIGNMENT/VALUE CHECK
        # Host: [k_type_dim1 (1), k_type_dim2 (1), hmm_1 (100)]
        host_data = f_x[keys.HOST_VEC][0]
        assert np.array_equal(host_data, [1, 1, 100])

        # INDEX CHECK: Since we have 1 phage and 1 host, the map is (0, 0)
        assert np.array_equal(f_x[keys.PAIR_INDICES][0], [0, 0])

    with h5py.File(cl_file, "r") as f_cl:
        # One valid interaction was found
        assert f_cl[keys.LABELS].shape[0] == 1
        assert f_cl[keys.LABELS][0] == 1


def test_build_model_data_inference_mode(setup_test_db, tmp_path):
    """Verifies that -1 labels are preserved during inference."""
    path = tmp_path / "inference.tsv"
    data = {"h_ready": {"p_ready": -1}}
    pd.DataFrame(data).T.to_csv(path, sep="\t")

    out_dir = tmp_path / "inference_output"
    base_name = "inf_run"
    keys = H5DataKeys()

    build_model_data(
        interaction_tsv=str(path),
        db_path=setup_test_db,
        output_dir=str(out_dir),
        base_filename=base_name,
        phage_table="phage_feats",
        host_table="host_feats",
        serotype_table="serotypes",
        is_inference=True,
    )

    with h5py.File(out_dir / f"cl_{base_name}.h5", "r") as f_cl:
        # Inference mode should not drop the -1 label
        assert f_cl[keys.LABELS].shape[0] == 1
        assert f_cl[keys.LABELS][0] == -1
