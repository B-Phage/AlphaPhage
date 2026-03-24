from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from models.inference import TransformerInferencer
from omegaconf import OmegaConf


@pytest.fixture
def mock_cfg():
    """The config passed to the Inferencer constructor."""
    return OmegaConf.create(
        {"dataset_name_host": "host_vec_1", "dataset_name_phage": "phage_vec_1", "batch_size": 2}
    )


@pytest.fixture
def mock_model():
    """Creates a mock LightningModule that mimics TransformerX behavior."""
    model = MagicMock()

    # Use a real dict here so we catch "dot vs bracket" errors
    model.hparams = {"cfg": {"batch_size": 2, "dim": 8}, "length_padded": 100, "num_tokens": 20}

    # model.cfg still needs to be an OmegaConf object for dot-access in predict()
    model.cfg = OmegaConf.create(
        {"dataset_name_host": "host_vec_1", "dataset_name_phage": "phage_vec_1", "batch_size": 2}
    )

    model.return_value = torch.tensor([[1.0, -1.0], [-1.0, 1.0]])
    return model


@patch("models.transformer_module.TransformerX.load_from_checkpoint")
def test_inferencer_init(mock_load, mock_model, mock_cfg):
    mock_load.return_value = mock_model

    inferencer = TransformerInferencer("fake_path.ckpt", mock_cfg)

    assert inferencer.expected_len == 100
    assert inferencer.expected_vocab == 20
    mock_load.assert_called_once()


@patch("models.inference.get_token_basics")
@patch("models.transformer_module.TransformerX.load_from_checkpoint")
def test_validate_hdf5_success(mock_load, mock_get_basics, mock_model, mock_cfg):
    mock_load.return_value = mock_model
    # Simulate HDF5 matching model exactly (len=100, vocab=20)
    mock_get_basics.return_value = (100, 20, [])

    inferencer = TransformerInferencer("fake_path.ckpt", mock_cfg)
    # Should not raise any error
    inferencer.validate_hdf5("fake_data.h5")


@patch("models.inference.get_token_basics")
@patch("models.transformer_module.TransformerX.load_from_checkpoint")
def test_validate_hdf5_mismatch_length(mock_load, mock_get_basics, mock_model, mock_cfg):
    mock_load.return_value = mock_model
    # HDF5 has length 200, but model expects 100
    mock_get_basics.return_value = (200, 20, [])

    inferencer = TransformerInferencer("fake_path.ckpt", mock_cfg)
    with pytest.raises(ValueError, match="Model expects length 100"):
        inferencer.validate_hdf5("fake_data.h5")


@patch("models.inference.get_token_basics")
@patch("models.transformer_module.TransformerX.load_from_checkpoint")
def test_validate_hdf5_mismatch_vocab(mock_load, mock_get_basics, mock_model, mock_cfg):
    mock_load.return_value = mock_model
    # HDF5 has token index 50, but model only knows up to 20
    mock_get_basics.return_value = (100, 50, [])

    inferencer = TransformerInferencer("fake_path.ckpt", mock_cfg)
    with pytest.raises(ValueError, match="contains token index 49"):
        inferencer.validate_hdf5("fake_data.h5")


@patch("models.inference.BinaryPairkeyFeatureDataset")
@patch("models.inference.DataLoader")
@patch("models.inference.get_token_basics")
@patch("models.transformer_module.TransformerX.load_from_checkpoint")
def test_predict_flow(mock_load, mock_get_basics, mock_loader, mock_ds, mock_model, mock_cfg):
    # Setup mocks
    mock_load.return_value = mock_model
    mock_get_basics.return_value = (100, 20, [])

    # Simulate a single batch of data from the DataLoader
    mock_batch = (torch.zeros((2, 100), dtype=torch.long), torch.zeros(2))
    mock_loader.return_value = [mock_batch]

    # Initialize and run
    inferencer = TransformerInferencer("fake_path.ckpt", mock_cfg)
    probs = inferencer.predict("fake_features.h5", "fake_labels.h5")

    # Assertions
    assert isinstance(probs, np.ndarray)
    assert probs.shape == (2, 2)
    # Ensure row sums are ~1.0 (validating softmax application)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)
