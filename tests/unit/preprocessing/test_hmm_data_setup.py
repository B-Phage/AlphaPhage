from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest
import yaml
from external_db_provisioning.hmm_data_setup import main, setup_hmm


@pytest.fixture
def base_entry():
    """Standard HMM entry for testing."""
    return {
        "name": "test.hmm",
        "dest_dir": "data/hmm",
        "source_url": "http://example.com/test.hmm",
        "final_sha256": "fake_hash",
        "can_split": True,
    }


# --- setup_hmm Tests ---


@patch("external_db_provisioning.hmm_data_setup.provision_asset")
@patch("external_db_provisioning.hmm_data_setup.split_hmms")
def test_setup_hmm_triggers_split_when_updated(mock_split, mock_provision, tmp_path, base_entry):
    # Setup paths
    base_entry["dest_dir"] = str(tmp_path)
    mock_provision.return_value = True  # Simulate a new download/update

    setup_hmm(base_entry)

    mock_provision.assert_called_once()
    # Should trigger split because was_updated is True
    mock_split.assert_called_once_with(str(tmp_path / "test.hmm"), str(tmp_path / "split"))


@patch("external_db_provisioning.hmm_data_setup.provision_asset")
@patch("external_db_provisioning.hmm_data_setup.split_hmms")
def test_setup_hmm_triggers_split_when_dir_empty(mock_split, mock_provision, tmp_path, base_entry):
    base_entry["dest_dir"] = str(tmp_path)
    mock_provision.return_value = False  # File already existed and was valid

    # split_dir exists but is empty
    split_dir = tmp_path / "split"
    split_dir.mkdir()

    setup_hmm(base_entry)

    # Should trigger split because the directory is empty even if not updated
    mock_split.assert_called_once()


@patch("external_db_provisioning.hmm_data_setup.provision_asset")
@patch("external_db_provisioning.hmm_data_setup.split_hmms")
def test_setup_hmm_skips_split_when_valid_and_not_updated(
    mock_split, mock_provision, tmp_path, base_entry
):
    base_entry["dest_dir"] = str(tmp_path)
    mock_provision.return_value = False

    # Simulate existing split files
    split_dir = tmp_path / "split"
    split_dir.mkdir()
    (split_dir / "model1.hmm").write_text("dummy")

    setup_hmm(base_entry)

    # Should NOT trigger split
    mock_split.assert_not_called()


@patch("external_db_provisioning.hmm_data_setup.provision_asset")
@patch("external_db_provisioning.hmm_data_setup.split_hmms")
def test_setup_hmm_no_split_allowed(mock_split, mock_provision, tmp_path, base_entry):
    base_entry["dest_dir"] = str(tmp_path)
    base_entry["can_split"] = False
    mock_provision.return_value = True

    setup_hmm(base_entry)

    mock_split.assert_not_called()


# --- CLI / Main Tests ---


def test_main_handles_list_config():
    mock_conf = [
        {"name": "hmm1", "dest_dir": "dir1", "source_url": "u1", "final_sha256": "h1"},
        {"name": "hmm2", "dest_dir": "dir2", "source_url": "u2", "final_sha256": "h2"},
    ]
    yaml_content = yaml.dump(mock_conf)

    with (
        patch("builtins.open", mock_open(read_data=yaml_content)),
        patch("argparse.ArgumentParser.parse_args", return_value=MagicMock(config="fake.yaml")),
        patch("external_db_provisioning.hmm_data_setup.setup_hmm") as mock_setup,
    ):

        main()
        assert mock_setup.call_count == 2


def test_setup_hmm_missing_required_keys():
    incomplete_entry = {"name": "test.hmm"}  # Missing dest_dir
    with pytest.raises(KeyError):
        setup_hmm(incomplete_entry)
