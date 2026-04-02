import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from packages.utils.stage_configs import run_cli, stage_runtime_configs


@pytest.fixture
def params_file(tmp_path):
    """Creates a dummy params.yaml and returns the path."""
    p = tmp_path / "params.yaml"
    return p


@pytest.fixture
def source_config(tmp_path):
    """Creates a source config file with some initial data."""
    src_dir = tmp_path / "ml_assets" / "config"
    src_dir.mkdir(parents=True)
    conf = src_dir / "test_v1.yaml"
    conf.write_text(yaml.dump({"param1": "value1", "param2": 50}))
    return conf


def test_stage_runtime_configs_success(params_file, source_config, tmp_path):
    # 1. Setup params.yaml content
    runtime_path = tmp_path / "ml_assets" / "runtime" / "test_runtime.yaml"
    params_data = {"test_conf": {"source": str(source_config), "runtime": str(runtime_path)}}
    params_file.write_text(yaml.dump(params_data))

    # 2. Execute
    stage_runtime_configs(str(params_file))

    # 3. Assertions
    assert runtime_path.exists()

    with open(runtime_path, "r") as f:
        hydrated_data = yaml.safe_load(f)

    assert hydrated_data["param1"] == "value1"
    assert hydrated_data["param2"] == 50
    # Verify the injection worked
    assert hydrated_data["self_path"] == str(runtime_path)


def test_stage_runtime_configs_missing_params_raises_error():
    with pytest.raises(FileNotFoundError):
        stage_runtime_configs("non_existent_params.yaml")


def test_stage_runtime_configs_missing_source_skips(params_file, tmp_path):
    # Point to a source that doesn't exist
    runtime_path = tmp_path / "runtime.yaml"
    params_data = {"missing_conf": {"source": "ghost_file.yaml", "runtime": str(runtime_path)}}
    params_file.write_text(yaml.dump(params_data))

    # Should not raise error, just skip
    stage_runtime_configs(str(params_file))
    assert not runtime_path.exists()


def test_stage_runtime_configs_creates_nested_directories(params_file, source_config, tmp_path):
    # Test that it creates multiple levels of directories for the runtime path
    deep_runtime = tmp_path / "level1" / "level2" / "level3" / "final.yaml"
    params_data = {"deep_conf": {"source": str(source_config), "runtime": str(deep_runtime)}}
    params_file.write_text(yaml.dump(params_data))

    stage_runtime_configs(str(params_file))
    assert deep_runtime.exists()


def test_stage_runtime_configs_ignores_non_config_blocks(params_file, tmp_path):
    # Test that blocks without source/runtime keys are ignored
    runtime_path = tmp_path / "runtime.yaml"
    params_data = {"random_param": 123, "other_block": {"just_a_key": "just_a_value"}}
    params_file.write_text(yaml.dump(params_data))

    # Should run without error and do nothing
    stage_runtime_configs(str(params_file))
    assert not runtime_path.exists()


def test_run_cli_success(params_file, source_config, tmp_path, monkeypatch):
    """Tests the CLI entry point with a valid params file."""
    runtime_path = tmp_path / "cli_runtime.yaml"
    params_data = {"cli_test": {"source": str(source_config), "runtime": str(runtime_path)}}
    params_file.write_text(yaml.dump(params_data))

    # Simulate: python script.py --params path/to/params.yaml
    monkeypatch.setattr(sys, "argv", ["stage_configs.py", "--params", str(params_file)])

    # Execution should not raise SystemExit(1)
    run_cli()

    assert runtime_path.exists()


def test_run_cli_failure(monkeypatch):
    """Tests that CLI exits with code 1 upon failure (e.g., missing params)."""
    monkeypatch.setattr(sys, "argv", ["stage_configs.py", "--params", "non_existent.yaml"])

    # argparse/run_cli should trigger the try-except and call exit(1)
    with pytest.raises(SystemExit) as e:
        run_cli()

    assert e.value.code == 1


# --- Error Block Coverage ---


def test_stage_runtime_configs_write_exception(params_file, source_config, tmp_path):
    """
    Covers the 'except Exception as e' block when writing to the destination fails.
    """
    runtime_path = tmp_path / "fail_to_write.yaml"
    params_data = {"fail_test": {"source": str(source_config), "runtime": str(runtime_path)}}
    params_file.write_text(yaml.dump(params_data))

    # We mock yaml.dump to raise an IOError when called,
    # simulating a disk failure or permission issue during the write phase.
    with patch("yaml.dump", side_effect=Exception("Disk Full")):
        with pytest.raises(Exception) as excinfo:
            stage_runtime_configs(str(params_file))

        assert "Disk Full" in str(excinfo.value)


# --- Edge Case: Empty Source ---


def test_stage_runtime_configs_empty_source(params_file, tmp_path):
    """Covers 'yaml.safe_load(f) or {}' when the source file is empty."""
    src = tmp_path / "empty.yaml"
    src.write_text("")  # Completely empty file

    dst = tmp_path / "dst.yaml"
    params_data = {"empty_test": {"source": str(src), "runtime": str(dst)}}
    params_file.write_text(yaml.dump(params_data))

    stage_runtime_configs(str(params_file))

    with open(dst, "r") as f:
        data = yaml.safe_load(f)

    # Should only contain the injected self_path
    assert data == {"self_path": str(dst)}
