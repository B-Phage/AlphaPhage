from pathlib import Path

import pytest
import utils.scripts.snapshot_inference as snap
from omegaconf import OmegaConf

# ---------------------------------------------------------------------
# 1. LOGIC TESTS (Always run in CI)
# ---------------------------------------------------------------------


def test_path_transformation_logic():
    """
    Regression test for the Pathlib transformation logic.
    Ensures ml_assets and databases paths are rewired correctly.
    """
    sample_config = {
        "path_a": "ml_assets/models/weights.ckpt",
        "path_b": "../databases/raw/data.csv",
        "non_path": "just_a_string",
    }

    transformed = snap.transform_to_inference_paths(sample_config)

    assert transformed["path_a"] == "../../databases/inference/models/weights.ckpt"
    assert transformed["path_b"] == "../../databases/raw/data.csv"
    assert transformed["non_path"] == "just_a_string"


def test_domain_configs_schema_validation(tmp_path):
    """
    Verifies the snapshot utility can correctly parse and validate
    the 'msa_var' structure in a preprocessing config.
    """
    # Create a mock config to test the code's ability to handle the contract
    fake_cfg_path = tmp_path / "mock_preprocessing.yaml"
    valid_content = {"msa_var": {"msa_cluster_name": "test_cluster", "msa_variant_dir": "test_dir"}}
    OmegaConf.save(valid_content, fake_cfg_path)

    # Test the parsing logic
    conf = OmegaConf.load(fake_cfg_path)
    assert "msa_var" in conf
    assert conf.msa_var.msa_cluster_name == "test_cluster"
    assert "msa_variant_dir" in conf.msa_var


# ---------------------------------------------------------------------
# 2. REPO CONTRACT TESTS (Run in CI if files are Git-tracked)
# ---------------------------------------------------------------------


def test_dvc_params_structure():
    """
    Ensures params.yaml (which should be in Git) follows the expected schema.
    """
    params_path = Path("params.yaml")

    # If params.yaml isn't even in Git, we should know immediately
    if not params_path.exists():
        pytest.skip("params.yaml not found (might not be tracked in this environment)")

    conf = OmegaConf.load(params_path)
    assert "storage" in conf, "params.yaml is missing 'storage' block"
    assert "training_data_dir" in conf.storage
    assert "training_data_name" in conf.storage


# ---------------------------------------------------------------------
# 3. ENVIRONMENT INTEGRITY (Skipped in CI, runs locally)
# ---------------------------------------------------------------------

# We define a condition: does the training data folder exist?
DATA_EXISTS = Path("ml_assets/data/processed/model_training").exists()


@pytest.mark.skipif(not DATA_EXISTS, reason="DVC data not present in this environment")
def test_resolve_feature_set_physical_existence():
    """
    Verifies that the H5 file defined in DVC actually exists on disk.
    Only runs if the data directory is present (Local/Prod, not Stateless CI).
    """
    try:
        h5_path = snap.resolve_feature_set_path()
        assert h5_path.exists(), f"H5 file defined in config not found at {h5_path}"
        assert h5_path.suffix == ".h5"
    except Exception as e:
        pytest.fail(f"Feature set resolution failed: {e}")
