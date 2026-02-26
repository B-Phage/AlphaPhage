import os
import shutil
from pathlib import Path

import pytest
import yaml

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
    Tests the full pipeline from .csa inputs to .mva/.mcd outputs.
    """
    # Update the HASH_STORE_DIR in the script to use our temp one
    # This prevents the test from writing to your real ml_assets folder
    monkeypatch.setattr(
        "run_variant_generation.HASH_STORE_DIR", Path(setup_test_env["config"]).parent / "hashes"
    )

    # Execute workflow
    run_variant_workflow(
        config_path=setup_test_env["config"], code_hash=[setup_test_env["hash_file"]], force=True
    )

    # --- Assertions ---
    out_dir = setup_test_env["output_dir"]

    # 1. Check if files exist for both inputs
    for phrog in ["phrog_1", "phrog_2"]:
        assert (out_dir / f"{phrog}.mva").exists(), f"MVA file for {phrog} missing"
        assert (out_dir / f"{phrog}.mcd").exists(), f"MCD file for {phrog} missing"

    # 2. Verify manifest creation
    assert setup_test_env["manifest"].exists(), "Manifest file was not created"

    # 3. Check content validity (non-empty files)
    with open(out_dir / "phrog_1.mva", "r") as f:
        content = f.read()
        assert len(content) > 0
        assert "phrog_1" in content or "," in content  # Basic CSV check
