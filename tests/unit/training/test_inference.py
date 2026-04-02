from pathlib import Path
from unittest.mock import MagicMock, patch

import h5py
import numpy as np
import pandas as pd
import pytest
import torch
from models.inference import TransformerInferencer
from omegaconf import OmegaConf


@pytest.fixture
def mock_cfg():
    return OmegaConf.create(
        {"dataset_name_host": "host_data", "dataset_name_phage": "phage_data", "batch_size": 2}
    )


@pytest.fixture
def mock_model(mock_cfg):
    """Creates a mock model that looks like a LightningModule."""
    model = MagicMock()
    model.hparams = {"length_padded": 10, "num_tokens": 100}
    model.cfg = mock_cfg
    # Mock the forward pass to return dummy logits
    model.return_value = torch.randn(2, 2)
    return model


@pytest.fixture
def temp_h5(tmp_path, mock_cfg):
    """Creates a valid dummy HDF5 file for testing."""
    h5_path = tmp_path / "test_features.h5"
    with h5py.File(h5_path, "w") as f:
        # 4 + 6 = 10 (Matches expected_len)
        f.create_dataset(mock_cfg.dataset_name_host, data=np.zeros((4, 4), dtype=int))
        f.create_dataset(mock_cfg.dataset_name_phage, data=np.zeros((4, 6), dtype=int))
    return str(h5_path)


## --- Initialization Tests ---


@patch("models.inference.TransformerX.load_from_checkpoint")
def test_init_success(mock_load, mock_model, mock_cfg):
    mock_load.return_value = mock_model

    inferencer = TransformerInferencer("fake.ckpt", mock_cfg)

    assert inferencer.expected_len == 10
    assert inferencer.expected_vocab == 100
    mock_model.eval.assert_called_once()
    mock_model.to.assert_called_once()


## --- Validation Logic Tests ---


@patch("models.inference.TransformerX.load_from_checkpoint")
def test_validate_hdf5_mismatch_length(mock_load, mock_model, mock_cfg, tmp_path):
    mock_load.return_value = mock_model
    inferencer = TransformerInferencer("fake.ckpt", mock_cfg)

    # Create H5 with wrong width (5+3=8 instead of 10)
    bad_h5 = tmp_path / "bad_len.h5"
    with h5py.File(bad_h5, "w") as f:
        f.create_dataset(mock_cfg.dataset_name_host, data=np.zeros((1, 5)))
        f.create_dataset(mock_cfg.dataset_name_phage, data=np.zeros((1, 3)))

    with pytest.raises(ValueError, match="DIMENSION MISMATCH"):
        inferencer.validate_hdf5(str(bad_h5))


@patch("models.inference.TransformerX.load_from_checkpoint")
def test_validate_hdf5_mismatch_vocab(mock_load, mock_model, mock_cfg, tmp_path):
    mock_load.return_value = mock_model
    inferencer = TransformerInferencer("fake.ckpt", mock_cfg)

    # Create H5 with token ID 150 (Model only supports up to 99)
    bad_h5 = tmp_path / "bad_vocab.h5"
    with h5py.File(bad_h5, "w") as f:
        f.create_dataset(mock_cfg.dataset_name_host, data=np.array([[150, 0, 0, 0, 0]]))
        f.create_dataset(mock_cfg.dataset_name_phage, data=np.zeros((1, 5)))

    with pytest.raises(ValueError, match="VOCAB MISMATCH"):
        inferencer.validate_hdf5(str(bad_h5))


## --- Prediction Tests ---


@patch("models.inference.TransformerX.load_from_checkpoint")
@patch("models.inference.BinaryPairkeyFeatureDataset")
@patch("models.inference.DataLoader")
def test_predict_flow(mock_loader_cls, mock_dataset_cls, mock_load, mock_model, mock_cfg, temp_h5):
    mock_load.return_value = mock_model
    inferencer = TransformerInferencer("fake.ckpt", mock_cfg)

    # Mock the DataLoader to return one batch
    # x shape: (batch_size, sequence_length)
    mock_batch = (torch.zeros((2, 10), dtype=torch.long), torch.zeros(2))
    mock_loader_cls.return_value = [mock_batch]

    # Run prediction
    probs = inferencer.predict(temp_h5, "fake_labels.csv")

    assert isinstance(probs, np.ndarray)
    assert probs.shape[0] == 2  # Batch size
    mock_model.assert_called_once()  # Verify forward pass ran


@patch("models.inference.TransformerX.load_from_checkpoint")
def test_predict_calls_validate(mock_load, mock_model, mock_cfg, temp_h5):
    """Ensure that predict internally calls validate_hdf5."""
    mock_load.return_value = mock_model
    inferencer = TransformerInferencer("fake.ckpt", mock_cfg)

    # Create a dummy batch to satisfy the loop and concatenate
    dummy_batch = (torch.zeros((1, 10)), torch.zeros(1))

    with (
        patch.object(inferencer, "validate_hdf5") as mock_val,
        patch("models.inference.BinaryPairkeyFeatureDataset"),
        patch("models.inference.DataLoader", return_value=[dummy_batch]),
    ):

        # We also need to mock the model's forward pass if we let the loop run
        mock_model.return_value = torch.randn(1, 2)

        inferencer.predict(temp_h5, "fake_labels.csv")

        # Verify validation was called before dataset/loader logic
        mock_val.assert_called_once_with(temp_h5)


@patch("models.inference.TransformerX.load_from_checkpoint")
def test_predict_empty_loader(mock_load, mock_model, mock_cfg, temp_h5):
    mock_load.return_value = mock_model
    inferencer = TransformerInferencer("fake.ckpt", mock_cfg)

    with (
        patch("models.inference.DataLoader", return_value=[]),
        patch("models.inference.BinaryPairkeyFeatureDataset"),
    ):

        # If you added the 'if not all_probs' check above:
        res = inferencer.predict(temp_h5, "fake_labels.csv")
        assert len(res) == 0
