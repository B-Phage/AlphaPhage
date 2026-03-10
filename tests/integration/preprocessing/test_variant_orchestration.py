import shutil
from pathlib import Path

import pandas as pd
import pytest
import utils.consol_log as log
import yaml

logger = log.get_logger()

from workflows.create_msa_variants import run_variant_workflow


@pytest.fixture
def setup_test_env(tmp_path):
    """
    Sets up a temporary directory structure and a mock-config file
    pointing to real test data.
    """
    # 1. Create directory structure
    input_dir = tmp_path / "inputs"
    output_base = tmp_path / "outputs"
    hash_dir = tmp_path / "hashes"
    input_dir.mkdir()
    output_base.mkdir()
    hash_dir.mkdir()

    # 2. Copy your real world .csa files to the temp input dir
    # Assuming these exist in your project's tests/data/
    source_data = Path("tests/data")
    for phrog in ["phrog_1.csa", "phrog_2.csa"]:
        shutil.copy(source_data / phrog, input_dir / phrog)

    # 3. Create a temporary config.yaml
    config_dict = {
        "general": {"name": "integration_test", "code_version": "1.0.0"},
        "msa_config": {"msa_csa_dir": str(input_dir)},
        "msa_var": {
            "msa_variant_dir": str(output_base),
            "msa_cluster_name": "test_run",
            "cluster_method": 2,  # Small number for fast test
            "dis_mat_method": "identity",
        },
    }

    config_path = tmp_path / "test_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config_dict, f)

    # We also need a dummy code_hash file to satisfy the workflow requirements
    hash_file = tmp_path / "code.hash"
    hash_file.write_text("dummy_hash_content")

    return {
        "config": str(config_path),
        "hash_file": hash_file,
        "output_dir": output_base / "test_run",
        "manifest": output_base / "test_run_manifest.json",
    }


def test_variant_workflow_end_to_end(setup_test_env, monkeypatch):
    """
    Tests the full pipeline from .csa inputs to .parquet shard outputs.
    """
    # Use an absolute path for the hash store to avoid permission issues in temp dirs
    hash_path = Path(setup_test_env["config"]).parent / "hashes"
    hash_path.mkdir(exist_ok=True)

    monkeypatch.setattr(
        "workflows.create_msa_variants.HASH_STORE_DIR",
        hash_path,
    )

    # Execute workflow
    run_variant_workflow(
        config_path=setup_test_env["config"], code_hash=[setup_test_env["hash_file"]], force=True
    )

    # --- Assertions ---
    out_dir = setup_test_env["output_dir"]

    # 1. Check if parquet shards exist for both inputs
    for phrog in ["phrog_1", "phrog_2"]:
        shard_file = out_dir / f"{phrog}.parquet"
        assert shard_file.exists(), f"Parquet shard for {phrog} missing in {out_dir}"

    # 2. Verify manifest creation
    assert setup_test_env["manifest"].exists(), "Manifest file was not created"

    # 3. Check content validity (Parquet files are binary, so we check if they are readable)
    # Using pandas to read the shard back
    df = pd.read_parquet(out_dir / "phrog_1.parquet")
    assert not df.empty
    assert "organism_id" in df.columns
    assert "variant_id" in df.columns
    assert (df["hmm_id"] == "phrog_1").all()
