from pathlib import Path

import pytest
import utils.scripts.snapshot_inference as snap
from omegaconf import OmegaConf


def test_dvc_params_contract():
    """
    Ensures params.yaml contains the 'storage' block
    required by the snapshot utility.
    """
    params_path = Path("params.yaml")
    assert params_path.exists(), "params.yaml is missing from repo root"

    conf = OmegaConf.load(params_path)
    assert "storage" in conf, "params.yaml is missing 'storage' block"
    assert "training_data_dir" in conf.storage
    assert "training_data_name" in conf.storage


def test_resolve_feature_set_exists():
    """
    Verifies that the H5 file defined in DVC actually exists on disk.
    This catches cases where DVC hasn't been pulled or paths changed.
    """
    try:
        h5_path = snap.resolve_feature_set_path()
        assert h5_path.exists(), f"H5 file defined in config not found at {h5_path}"
        assert h5_path.suffix == ".h5"
    except Exception as e:
        pytest.fail(f"Feature set resolution failed: {e}")


def test_domain_configs_contract():
    """
    Checks that every domain preprocessing config mentioned in the
    pipeline has the 'msa_var' structure needed for gallery porting.
    """
    runtime_cfg_dir = Path("ml_assets/runtime_configs")
    domain_cfgs = list(runtime_cfg_dir.glob("*_preprocessing.yaml"))

    assert len(domain_cfgs) > 0, "No domain configs found in ml_assets/runtime_configs"

    for cfg_path in domain_cfgs:
        conf = OmegaConf.load(cfg_path)
        # These are the keys the snapshot utility 'reaches' for
        assert "msa_var" in conf, f"{cfg_path.name} missing 'msa_var' block"
        assert "msa_cluster_name" in conf.msa_var
        assert "msa_variant_dir" in conf.msa_var


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
