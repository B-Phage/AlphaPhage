import os
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import h5py
import numpy as np
from omegaconf import OmegaConf
from training.train_utils import (
    get_datamodule,
    get_experiment_tracker_method,
    get_model,
    get_token_basics,
)


class TestTrainUtils(unittest.TestCase):

    def setUp(self):
        """Set up mock paths and basic configs."""
        self.test_h5 = "test_features.h5"
        self.host_ds = "host_data"
        self.phage_ds = "phage_data"

        # Create a real small H5 file for integration testing the logic
        with h5py.File(self.test_h5, "w") as f:
            # Create two samples of length 5
            f.create_dataset(
                self.host_ds, data=np.array([[1, 2, 3, 0, 0], [1, 2, 4, 0, 0]], dtype=np.int32)
            )
            f.create_dataset(
                self.phage_ds, data=np.array([[5, 6, 7, 0, 0], [5, 6, 8, 0, 0]], dtype=np.int32)
            )

    def tearDown(self):
        """Clean up the test H5 file."""
        if os.path.exists(self.test_h5):
            os.remove(self.test_h5)

    # --- 1. Data Contract Tests ---

    def test_get_token_basics_valid(self):
        """MLOps: Verify token extraction correctly identifies vocab range and padding."""
        length_padded, num_tokens, tokens_in_data = get_token_basics(
            self.test_h5, self.host_ds, self.phage_ds
        )

        # Combined length of first pair (5 + 5)
        self.assertEqual(length_padded, 10)
        # Unique tokens in dummy data are [0, 1, 2, 3, 4, 5, 6, 7, 8]
        # num_tokens should be max(tokens) + 1 = 9
        self.assertEqual(num_tokens, 9)
        self.assertIn(8, tokens_in_data)
        self.assertIn(0, tokens_in_data)

    # --- 2. Factory Pattern Tests ---

    def test_get_experiment_tracker_method_dvclive(self):
        """Ensure DVCLiveLogger is instantiated with correct MLOps flags."""
        cfg = OmegaConf.create({"experiment_tracker_method": "DVCLiveLogger"})
        mock_dir = "test_logs"

        # Patching to avoid actual directory creation during test
        with patch("dvclive.lightning.DVCLiveLogger") as mock_logger:
            get_experiment_tracker_method(cfg, mock_dir)
            mock_logger.assert_called_once_with(
                dir=mock_dir, log_model=False, dvcyaml=False, report="html"
            )

    def test_get_model_transformer_x(self):
        """Verify the TransformerX factory correctly passes architecture specs."""
        cfg = OmegaConf.create(
            {
                "architecture": "TransformerX",
                "dim": 8,
                "dim1": 8,
                "dim2": 8,
                "heads": 2,
                "mult": 2,
                "depth": 1,
                "archspec": "some_key",
                "archspec2": "some_key2",
                "lcom": None,
                "lr": 0.001,
            }
        )

        # Mocking the actual model class to verify the call signature
        with patch("models.transformer_module.TransformerX") as mock_model:
            get_model(cfg, num_tokens=10, vocab_size_out=2, length_padded=100)

            # Check if it was called with our provided dimensions
            args, _ = mock_model.call_args
            self.assertEqual(args[1], 10)  # num_tokens
            self.assertEqual(args[3], 100)  # length_padded

    # --- 3. Error Handling (Guardrail) Tests ---

    def test_invalid_configs_raise_errors(self):
        """MLOps: Ensure the system fails fast on unsupported configurations."""
        bad_cfg = OmegaConf.create({"architecture": "ChatGPT", "reader_mode": "csv"})

        with self.subTest("Unsupported Architecture"):
            with self.assertRaises(ValueError):
                get_model(bad_cfg, 10, 2, 100)

        with self.subTest("Unsupported Logger"):
            bad_cfg.experiment_tracker_method = "Tensorboard"
            with self.assertRaises(ValueError):
                get_experiment_tracker_method(bad_cfg, "logs")


class TestGetDatamodule(unittest.TestCase):
    """Note that there is a dedicated dataloaders module. This test is just to
    verify the factory function in train_utils.py."""

    def setUp(self):
        """Set up standard mock config."""
        self.cfg = OmegaConf.create(
            {
                "reader_mode": "binary_pairkey",
                "batch_size": 32,
                # Add any other minimal keys BinaryPairkeyDataModule might expect
            }
        )

    @patch("dataloaders.dataloaders.BinaryPairkeyDataModule")
    def test_get_datamodule_success(self, mock_dm_class):
        """MLOps: Verify the factory returns the correct DataModule instance."""
        # 1. Setup the mock instance
        mock_instance = MagicMock()
        mock_dm_class.return_value = mock_instance

        # 2. Call the factory
        result = get_datamodule(self.cfg)

        # 3. Assertions
        # Ensure the class was instantiated with the config
        mock_dm_class.assert_called_once_with(self.cfg)
        # Ensure the returned object is what we expected
        self.assertEqual(result, mock_instance)

    def test_get_datamodule_invalid_mode(self):
        """Guardrail: Ensure an unsupported reader_mode raises a ValueError."""
        # Change mode to something not implemented
        self.cfg.reader_mode = "unsupported_csv_mode"

        with self.assertRaises(ValueError) as cm:
            get_datamodule(self.cfg)

        self.assertIn("Unknown reader_mode: unsupported_csv_mode", str(cm.exception))

    def test_get_datamodule_missing_key(self):
        """Robustness: Ensure the function handles a missing reader_mode key."""
        empty_cfg = OmegaConf.create({})

        # Depending on how OmegaConf is configured, this might raise MissingMandatoryValue
        # or AttributeError. Standard dict-like access on DictConfig raises AttributeError if missing.
        with self.assertRaises(Exception):
            get_datamodule(empty_cfg)


if __name__ == "__main__":
    unittest.main()
