# tests/utils/test_config_loader.py
"""
To get the correct installed packages
python -m unittest tests.test_config_loader
PYTHONPATH=./packages pytest tests/
"""

import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import h5py
import numpy as np
import torch
from arc.TransformerX import TransformerX
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from training.train import main
from training.train_utils import get_token_basics


class TestTransformerX(unittest.TestCase):
    def setUp(self):
        """Set up a mock configuration similar to your config_test_2.yaml
        and mock the trainer.
        """

        self.num_tokens = 20
        self.vocab_size_out = 2
        self.length_padded = 128

        self.cfg = OmegaConf.create(
            {
                "seed": 42,
                "verboselevel": "ERROR",
                # "debug_mode" : False,
                "train_features_path": "fake_feat.h5",
                # "train_features_path" : "../../bp/AlphaX/proc/sim_data/Tests/x_Test_50_50_TOK_test_AT_cond2_aligned_test_binary_pairkey.h5",
                # "train_labels_path" : "../../bp/AlphaX/proc/sim_data/Tests/cl_Test_50_50_TOK_test_AT_cond2_aligned_test_binary_pairkey.h5",
                # "train_features_path" : "tests/data/x_Test_50_50_TOK_test_AT_cond2_aligned_test_binary_pairkey.h5",
                # "train_labels_path" : "tests/data/cl_Test_50_50_TOK_test_AT_cond2_aligned_test_binary_pairkey.h5",
                # "train_features_path" : "tests/data/x_Test_50_50_TOK_test_AT_cond2_unaligned_test_binary_pairkey.h5",
                # "train_labels_path" : "tests/data/cl_Test_50_50_TOK_test_AT_cond2_unaligned_test_binary_pairkey.h5",
                # "debug_features_path" : "",
                # "debug_labels_path" : "",
                # "pair_indices_name" : "pair_indices",
                "dataset_name_host": "host_vec_1",
                "dataset_name_phage": "phage_vec_1",
                "reader_mode": "binary_pairkey",
                "architecture": "TransformerX",
                "num_workers": 0,
                "trainer": {
                    "logger_method": "MLFlowLogger",
                    "project": "AlphaPhage_Testing",
                    "run_name": "test_run_2",
                    "mlflow_tracking_uri": None,
                    "epochs": 300,
                    "accelerator": "cpu",
                    "devices": 1,
                    "precision": 32,
                    "batch_size": 10,
                    "save_top_k": 3,
                    "patience": 5,
                    "val_fraction": 0.15,
                },
                "hyp": {
                    "archspec": "nope_nt_nonormpostl_sum",
                    "archspec2": "L4_bn",
                    "dim": 8,
                    "lcom": 30,
                    "dim1": 8,
                    "dim2": 8,
                    "heads": 4,
                    "mult": 4,
                    "depth": 1,
                    "vocab_size_out": 2,
                },
                "optimizer": {
                    "lr": 0.001,
                },
            }
        )
        self.model = TransformerX(
            self.cfg, self.num_tokens, self.vocab_size_out, self.length_padded
        )
        # Mock the log method directly on the instance
        self.model.log = MagicMock()
        # Manually attach a mock trainer to satisfy the log() requirement
        self.model.trainer = MagicMock()

    #################################
    # @patch("pytorch_lightning.Trainer")
    # @patch("training.train.get_token_basics")
    # @patch("hydra.core.hydra_config.HydraConfig.get") # Mock the global Hydra helper
    # def test_main_initialization(self, mock_hydra_get, mock_get_basics, mock_trainer):
    #     """Test if main() executes and initializes the trainer correctly."""

    #     # 1. Mock the specific Hydra call your main() uses:
    #     # main() calls: hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    #     mock_hydra_get.return_value.runtime.output_dir = "/tmp/test_output"

    #     # 2. Mock get_token_basics return values
    #     mock_get_basics.return_value = (128, 20, [0, 1])

    #     # 3. Mock the Trainer instance
    #     mock_trainer_instance = MagicMock()
    #     mock_trainer.return_value = mock_trainer_instance

    #     # 4. Use __wrapped__ to bypass the @hydra.main decorator
    #     main.__wrapped__(self.cfg)

    #     # 5. Assertions
    #     mock_get_basics.assert_called_once()
    #     mock_trainer_instance.fit.assert_called_once()

    @patch("pytorch_lightning.Trainer")
    @patch("training.train.get_token_basics")
    @patch("hydra.core.hydra_config.HydraConfig.get")
    def test_main_initialization(self, mock_hydra_get, mock_get_basics, mock_trainer):
        """Test if main() executes and initializes the trainer correctly."""

        # 1. Create a temporary directory for this specific test run
        with tempfile.TemporaryDirectory() as tmp_log_dir:

            # 2. Mock Hydra's output_dir to point to our temp directory
            mock_hydra_get.return_value.runtime.output_dir = tmp_log_dir

            # 3. Mock dependencies
            mock_get_basics.return_value = (128, 20, [0, 1])
            mock_trainer_instance = MagicMock()
            mock_trainer.return_value = mock_trainer_instance

            # 4. Run the main function logic
            main.__wrapped__(self.cfg)

            # 7. Verify Callbacks (EarlyStopping and ModelCheckpoint)
            args, kwargs = mock_trainer.call_args
            callbacks = kwargs.get("callbacks", [])

            # Check if the list contains the right types
            from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

            has_checkpoint = any(isinstance(c, ModelCheckpoint) for c in callbacks)
            has_early_stop = any(isinstance(c, EarlyStopping) for c in callbacks)

            self.assertTrue(has_checkpoint, "ModelCheckpoint callback missing from Trainer.")
            self.assertTrue(has_early_stop, "EarlyStopping callback missing from Trainer.")

            # 8. Verify Checkpoint logic (Optional: check the specific monitor)
            for c in callbacks:
                if isinstance(c, ModelCheckpoint):
                    self.assertEqual(c.monitor, "val_loss")
                if isinstance(c, EarlyStopping):
                    self.assertEqual(c.patience, self.cfg.trainer.patience)

            # 5. Assertions: Check if directories were actually created
            checkpoint_dir = os.path.join(tmp_log_dir, "checkpoints")
            self.assertTrue(os.path.exists(checkpoint_dir), "Checkpoint directory was not created.")

            # 6. Verify Trainer was called with the correct logger directory
            # Based on your script: logger_object = pl.loggers.CSVLogger(log_dir)
            mock_trainer.assert_called_once()
            _, kwargs = mock_trainer.call_args

            # This confirms the PL Trainer received the logger we expected
            self.assertIsNotNone(kwargs["logger"], "Trainer was not provided with a logger.")

    @patch("training.train.get_token_basics")
    @patch("hydra.core.hydra_config.HydraConfig.get")
    def test_unsupported_architecture(self, mock_hydra_get, mock_get_basics):
        # Mock Hydra again for this test
        mock_hydra_get.return_value.runtime.output_dir = "/tmp/test_output"

        self.cfg.architecture = "UnknownNet"
        mock_get_basics.return_value = (128, 20, [])

        with self.assertRaises(ValueError) as cm:
            main.__wrapped__(self.cfg)
        self.assertIn("Unknown architecture", str(cm.exception))

    ###########

    def test_forward_shape(self):
        """Test if the model produces the correct output shape."""
        # Create dummy input: batch_size=4, sequence_length=128
        dummy_input = torch.randint(0, self.num_tokens, (4, self.length_padded))

        output = self.model(dummy_input)

        # Assert output shape (Batch, Length, Vocab) or as per your architecture
        expected_shape = (4, self.vocab_size_out)
        self.assertEqual(output.shape, expected_shape)

    def test_training_step(self):
        """Test if training_step computes a loss."""
        batch = (
            torch.randint(0, self.num_tokens, (4, self.length_padded)),
            torch.randint(0, self.vocab_size_out, (4,)),
        )

        loss = self.model.training_step(batch, 0)

        self.assertIsInstance(loss, torch.Tensor)
        self.assertFalse(torch.isnan(loss))

    def test_parameter_update(self):
        """Verification that gradients actually flow and weights change."""
        optimizer = self.model.configure_optimizers()
        batch = (
            torch.randint(0, self.num_tokens, (4, self.length_padded)),
            torch.randint(0, self.vocab_size_out, (4,)),
        )

        # 'model.embed.weight' should always run for current TransformerX that
        # requires token inputs. Other layers depend on arcspec and arcpec2.

        # Initial weight
        initial_weight = self.model.state_dict()["model.embed.weight"].clone()
        loss = self.model.training_step(batch, 0)
        loss.backward()
        optimizer.step()

        # New weight
        updated_weight = self.model.state_dict()["model.embed.weight"]

        self.assertFalse(torch.equal(initial_weight, updated_weight))

    def test_training_step_all_arcspecs(self):
        """Test if training_step computes a loss for all archspec settings."""
        batch = (
            torch.randint(0, self.num_tokens, (4, self.length_padded)),
            torch.randint(0, self.vocab_size_out, (4,)),
        )
        archspec_list = [
            "rope_tt_normpostl_view",
            "rope_tt_normpostl_sum",
            "rope_tt_nonormpostl_view",
            "rope_tt_nonormpostl_sum",
            "rope_nt_normpostl_view",
            "rope_nt_normpostl_sum",
            "rope_nt_nonormpostl_view",
            "rope_nt_nonormpostl_sum",
            "nope_tt_normpostl_view",
            "nope_tt_normpostl_sum",
            "nope_tt_nonormpostl_view",
            "nope_tt_nonormpostl_sum",
            "nope_nt_normpostl_view",
            "nope_nt_normpostl_sum",
            "nope_nt_nonormpostl_view",
            "nope_nt_nonormpostl_sum",
        ]
        archspec2_list = [
            "L4_bn",
            "L4_ln",
            "L4_nn",
            "L3_bn",
            "L3_ln",
            "L3_nn",
            "L2_bn",
            "L2_ln",
            "L2_nn",
            "L1_nn",
        ]
        for archspec in archspec_list:
            for archspec2 in archspec2_list:
                with self.subTest(archspec=archspec, archspec2=archspec2):
                    # print(f'archspec: {archspec} archspec2: {archspec2}')
                    self.cfg.hyp.archspec = archspec
                    self.cfg.hyp.archspec2 = archspec2
                    new_model = TransformerX(
                        self.cfg, self.num_tokens, self.vocab_size_out, self.length_padded
                    )
                    new_model.log = MagicMock()
                    new_model.trainer = MagicMock()
                    loss = new_model.training_step(batch, 0)

                    self.assertIsInstance(loss, torch.Tensor)
                    self.assertFalse(torch.isnan(loss))


class TestTokenBasics(unittest.TestCase):
    def setUp(self):
        """Create a temporary HDF5 file for testing."""
        self.test_dir = tempfile.TemporaryDirectory()
        self.h5_path = os.path.join(self.test_dir.name, "test_data.h5")

        self.host_key = "host_data"
        self.phage_key = "phage_data"

        # Define mock data:
        # Host: 2 samples, length 3. Phage: 2 samples, length 2.
        # Tokens present: 0, 1, 5, 10
        self.host_data = np.array([[1, 0, 5], [0, 1, 1]], dtype=np.int32)
        self.phage_data = np.array([[10, 5], [0, 10]], dtype=np.int32)

        with h5py.File(self.h5_path, "w") as f:
            f.create_dataset(self.host_key, data=self.host_data)
            f.create_dataset(self.phage_key, data=self.phage_data)

    def tearDown(self):
        """Clean up the temporary directory."""
        self.test_dir.cleanup()

    def test_get_token_basics_values(self):
        """Test if the function correctly calculates length and tokens."""
        length_padded, num_tokens, tokens_in_data = get_token_basics(
            self.h5_path, self.host_key, self.phage_key
        )

        # 1. Check length_padded (3 + 2 = 5)
        self.assertEqual(length_padded, 5)

        # 2. Check unique tokens (should be [0, 1, 5, 10] sorted)
        expected_tokens = [0, 1, 5, 10]
        self.assertCountEqual(tokens_in_data, expected_tokens)

        # 3. Check num_tokens (max token is 10, so num_tokens = 11)
        self.assertEqual(num_tokens, 11)

    def test_get_token_basics_invalid_path(self):
        """Test how it handles a non-existent file."""
        with self.assertRaises(OSError):
            get_token_basics("non_existent_file.h5", "h", "p")


if __name__ == "__main__":
    unittest.main(exit=False)
