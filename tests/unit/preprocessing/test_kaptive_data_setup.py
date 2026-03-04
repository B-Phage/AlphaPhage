from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml
from external_db_provisioning.kaptive_data_setup import main, setup_assets

# --- setup_assets Tests ---


@patch("external_db_provisioning.kaptive_data_setup.provision_asset")
def test_setup_assets_list_config(mock_provision, tmp_path):
    """Test that setup_assets iterates correctly over a list of entries."""
    config_file = tmp_path / "kaptive_list.yaml"
    manifest = [
        {
            "name": "db1.gbk",
            "dest_dir": str(tmp_path),
            "source_url": "url1",
            "final_sha256": "hash1",
        },
        {
            "name": "db2.gbk",
            "dest_dir": str(tmp_path),
            "source_url": "url2",
            "final_sha256": "hash2",
        },
    ]
    config_file.write_text(yaml.dump(manifest))

    setup_assets(config_file)

    assert mock_provision.call_count == 2
    # Check that it combined dest_dir and name correctly
    args, kwargs = mock_provision.call_args_list[0]
    assert kwargs["target_path"] == tmp_path / "db1.gbk"


@patch("external_db_provisioning.kaptive_data_setup.provision_asset")
def test_setup_assets_single_dict_config(mock_provision, tmp_path):
    """Test that setup_assets handles a single dictionary (not in a list)."""
    config_file = tmp_path / "kaptive_single.yaml"
    manifest = {
        "name": "single.gbk",
        "dest_dir": str(tmp_path),
        "source_url": "url",
        "final_sha256": "hash",
    }
    config_file.write_text(yaml.dump(manifest))

    setup_assets(config_file)

    mock_provision.assert_called_once()
    _, kwargs = mock_provision.call_args
    assert kwargs["url"] == "url"


def test_setup_assets_missing_file():
    """Test that it raises FileNotFoundError if config doesn't exist."""
    with pytest.raises(FileNotFoundError):
        setup_assets(Path("non_existent_file.yaml"))


@patch("external_db_provisioning.kaptive_data_setup.provision_asset")
def test_setup_assets_missing_keys(mock_provision, tmp_path):
    """Test that it raises KeyError when required fields are missing."""
    config_file = tmp_path / "broken.yaml"
    manifest = {"name": "broken.gbk"}  # missing source_url and final_sha256
    config_file.write_text(yaml.dump(manifest))

    with pytest.raises(KeyError):
        setup_assets(config_file)


# --- CLI / Main Tests ---


@patch("external_db_provisioning.kaptive_data_setup.setup_assets")
@patch("argparse.ArgumentParser.parse_args")
def test_main_calls_setup_assets(mock_args, mock_setup, tmp_path):
    """Test that main correctly routes valid files to setup_assets."""
    config_file = tmp_path / "valid.yaml"
    config_file.write_text("dummy")

    mock_args.return_value = MagicMock(config=config_file)

    main()
    mock_setup.assert_called_once_with(config_file)


@patch("external_db_provisioning.kaptive_data_setup.logger")
@patch("argparse.ArgumentParser.parse_args")
def test_main_logs_error_on_missing_config(mock_args, mock_logger, tmp_path):
    """Test that main handles non-existent config files gracefully."""
    bad_path = tmp_path / "ghost.yaml"
    mock_args.return_value = MagicMock(config=bad_path)

    main()
    mock_logger.error.assert_called_with(f"Config file not found: {bad_path}")
