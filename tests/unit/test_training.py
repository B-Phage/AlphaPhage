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
from arc.TransformerX import TransformerX, archspecs, archspecs2
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from training.train import main
from training.train_utils import get_token_basics


class TestTrainerInitialization(unittest.TestCase):

    def setUp(self):
        # Clear Hydra state to ensure each test starts with a fresh 'initialize'
        GlobalHydra.instance().clear()

    @patch("pytorch_lightning.Trainer")
    @patch("training.train.get_token_basics")
    @patch("hydra.core.hydra_config.HydraConfig.get")
    def test_invalid_reader_mode(self, mock_hydra_get, mock_get_basics, mock_trainer):
        """Tests that an unknown reader_mode raises a ValueError."""

        # Setup mocks to allow the script to reach the reader_mode check
        mock_hydra_get.return_value.runtime.output_dir = "/tmp/test_output"
        mock_get_basics.return_value = (100, 20, 1000)

        with initialize(version_base="1.3", config_path="../../tests/unit/data"):
            cfg = compose(
                config_name="config_test_2",
                overrides=["reader_mode=invalid_mode", "verboselevel=40"],
            )

            with self.assertRaisesRegex(ValueError, "Unknown reader_mode"):
                main.__wrapped__(cfg)

        mock_trainer.assert_not_called()

    @patch("pytorch_lightning.Trainer")
    @patch("training.train.get_token_basics")
    @patch("hydra.core.hydra_config.HydraConfig.get")
    @patch("dataloaders.dataloaders.BinaryPairkeyDataModule")
    def test_invalid_architecture(self, mock_dm, mock_hydra_get, mock_get_basics, mock_trainer):
        """Tests that an unknown architecture raises a ValueError."""

        # Setup mocks
        mock_hydra_get.return_value.runtime.output_dir = "/tmp/test_output"
        mock_get_basics.return_value = (100, 20, 1000)

        # We must mock the DataModule because it's initialized BEFORE the architecture check
        mock_dm.return_value = MagicMock()

        with initialize(version_base="1.3", config_path="../../tests/unit/data"):
            cfg = compose(
                config_name="config_test_2",
                overrides=[
                    "reader_mode=binary_pairkey",  # Pass the first check
                    "architecture=UnknownModel",  # Fail the second check
                    "verboselevel=40",
                ],
            )

            with self.assertRaisesRegex(ValueError, "Unknown architecture"):
                main.__wrapped__(cfg)

        mock_trainer.assert_not_called()


class TestAllLoggers(unittest.TestCase):

    def setUp(self):
        GlobalHydra.instance().clear()

    @patch("pytorch_lightning.Trainer")
    @patch("training.train.get_token_basics")
    @patch("hydra.core.hydra_config.HydraConfig.get")
    @patch("dataloaders.dataloaders.BinaryPairkeyDataModule")
    @patch("arc.TransformerX.TransformerX")
    @patch("pytorch_lightning.loggers.WandbLogger")
    @patch("pytorch_lightning.loggers.CSVLogger")
    @patch("pytorch_lightning.loggers.TensorBoardLogger")
    @patch("pytorch_lightning.loggers.MLFlowLogger")
    @patch("dvclive.lightning.DVCLiveLogger")  # Added DVC just in case
    def test_logger_initialization_grid(
        self,
        mock_dvc,
        mock_mlflow,
        mock_tb,
        mock_csv,
        mock_wandb,
        mock_model,
        mock_dm,
        mock_hydra_get,
        mock_get_basics,
        mock_trainer,
    ):

        # 1. Setup global mocks
        mock_hydra_get.return_value.runtime.output_dir = "/tmp/test_output"
        mock_get_basics.return_value = (100, 20, 1000)

        # 2. Define the cases: (method_name, expected_mock, extra_overrides)
        logger_test_cases = [
            ("WandbLogger", mock_wandb, []),
            ("CSVLogger", mock_csv, []),
            ("TensorBoardLogger", mock_tb, []),
            ("DVCLiveLogger", mock_dvc, []),
            # MLFlow Case 1: Remote URI
            ("MLFlowLogger", mock_mlflow, ["trainer.mlflow_tracking_uri=http://localhost:5000"]),
            # MLFlow Case 2: Local Fallback (null)
            ("MLFlowLogger", mock_mlflow, ["trainer.mlflow_tracking_uri=null"]),
        ]

        # 3. Loop through cases using subTest
        for method_name, expected_mock, extra_overrides in logger_test_cases:
            with self.subTest(logger_method=method_name, extras=extra_overrides):
                # Reset all mocks
                for m in [mock_mlflow, mock_tb, mock_csv, mock_wandb, mock_dvc, mock_trainer]:
                    m.reset_mock()

                with initialize(version_base="1.3", config_path="../../tests/unit/data"):
                    # Build overrides list
                    base_overrides = [
                        f"trainer.logger_method={method_name}",
                        "trainer.run_name=test_run",
                        "verboselevel=40",
                    ]
                    final_overrides = base_overrides + extra_overrides

                    cfg = compose(config_name="config_test_2", overrides=final_overrides)
                    main.__wrapped__(cfg)

                # 4. Verify the correct logger was called
                expected_mock.assert_called_once()

                # 5. Specific check for MLFlow logic if that's the current test
                if method_name == "MLFlowLogger":
                    call_kwargs = expected_mock.call_args.kwargs
                    if "http" in str(extra_overrides):
                        self.assertEqual(call_kwargs["tracking_uri"], "http://localhost:5000")
                    else:
                        # Local fallback check
                        self.assertTrue(call_kwargs["tracking_uri"].startswith("file://"))

                # 6. Verify the Trainer received this specific logger instance
                passed_logger = mock_trainer.call_args.kwargs["logger"]
                self.assertEqual(passed_logger, expected_mock.return_value)

    @patch("training.train.get_token_basics")
    @patch("hydra.core.hydra_config.HydraConfig.get")
    def test_unknown_logger_error_message(self, mock_hydra_get, mock_get_basics):
        """Tests that a typo in the logger method raises the correct ValueError."""

        mock_hydra_get.return_value.runtime.output_dir = "/tmp/test_output"
        mock_get_basics.return_value = (100, 20, 1000)
        invalid_logger = "InvalidLogger"

        with initialize(version_base="1.3", config_path="../../tests/unit/data"):
            cfg = compose(
                config_name="config_test_2",
                overrides=[f"trainer.logger_method={invalid_logger}", "verboselevel=40"],
            )

            # Testing for the specific error message you wrote in your script
            with self.assertRaisesRegex(ValueError, f"Unknown logger method: {invalid_logger}"):
                main.__wrapped__(cfg)


class TestTransformerX(unittest.TestCase):
    def setUp(self):
        """Set up a mock configuration similar to your config_test_2.yaml
        and mock the trainer.
        """
        GlobalHydra.instance().clear()

        self.num_tokens = 20
        self.vocab_size_out = 2
        self.length_padded = 128

        self.cfg = OmegaConf.create(
            {
                "seed": 42,
                "verboselevel": "ERROR",
                "debug_mode": False,
                "train_features_path": "fake_feat.h5",
                "train_labels_path": "fake_label.h5",
                "debug_features_path": "fake_debug_feat.h5",
                "debug_labels_path": "fake_debug_label.h5",
                "pair_indices_name": "pair_indices",
                "dataset_name_host": "host_vec_1",
                "dataset_name_phage": "phage_vec_1",
                "reader_mode": "binary_pairkey",
                "architecture": "TransformerX",
                "num_workers": 0,
                "trainer": {
                    "logger_method": "DVCLiveLogger",
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
                    # "archspec": "nope_nt_nonormpostl_sum",
                    "archspec": "nope_nt_nonormpostl_cls",
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
            self.cfg,
            self.num_tokens,
            self.vocab_size_out,
            self.length_padded,
            archspecs,
            archspecs2,
        )
        # Mock the log method directly on the instance
        self.model.log = MagicMock()
        # Manually attach a mock trainer to satisfy the log() requirement
        self.model.trainer = MagicMock()

    #################################
    # Note: The order of the @patch decorators is the reverse of the arguments in
    # the function signature
    @patch("pytorch_lightning.Trainer")
    @patch("training.train.get_token_basics")
    @patch("hydra.core.hydra_config.HydraConfig.get")  # Mock the global Hydra helper
    def test_main_initialization(self, mock_hydra_get, mock_get_basics, mock_trainer):
        """Test if main() executes and initializes the trainer correctly."""

        # 1. Mock the specific Hydra call your main() uses:
        # main() calls: hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        # When the code asks for the output directory, just tell it it's in /tmp.
        mock_hydra_get.return_value.runtime.output_dir = "/tmp/test_output"

        # 2. Mock get_token_basics return values
        mock_get_basics.return_value = (128, 20, [0, 1])

        # 3. Mock the Trainer instance
        mock_trainer_instance = MagicMock()
        mock_trainer.return_value = mock_trainer_instance

        # 4. Use __wrapped__ to bypass the @hydra.main decorator
        # allows the test to reach inside the decorator and run just your Python
        # code, passing in the self.cfg object you created manually
        main.__wrapped__(self.cfg)

        # 5. Assertions
        mock_get_basics.assert_called_once()
        mock_trainer_instance.fit.assert_called_once()

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

    def test_validation_step(self):
        """Test if validation_step executes and logs."""
        batch = (
            torch.randint(0, self.num_tokens, (4, self.length_padded)),
            torch.randint(0, self.vocab_size_out, (4,)),
        )
        self.model.log.reset_mock()

        # Execute (will likely return None)
        self.model.validation_step(batch, 0)

        # Assert that logging was triggered (this proves the code ran)
        self.assertTrue(self.model.log.called)

    def test_test_step(self):
        """Test if test_step executes and logs."""
        batch = (
            torch.randint(0, self.num_tokens, (4, self.length_padded)),
            torch.randint(0, self.vocab_size_out, (4,)),
        )
        self.model.log.reset_mock()

        self.model.test_step(batch, 0)

        self.assertTrue(self.model.log.called)

    @patch("dataloaders.dataloaders.h5py.File")
    def test_datamodule_predict_stage(self, mock_h5):
        """Covers the 'predict' branch and handles HDF5 file requirements."""
        from dataloaders.dataloaders import BinaryPairkeyDataModule

        # 1. Setup the mock HDF5 structure
        # This simulates: f['data'] returning a list of length 10
        mock_file = mock_h5.return_value.__enter__.return_value
        mock_file.__getitem__.return_value = [0] * 10

        # 2. Initialize DataModule
        dm = BinaryPairkeyDataModule(self.cfg)

        # 3. Hit all conditional setup branches
        # This will now cover 'predict', 'fit', and 'test' in setup()
        dm.setup(stage="predict")
        self.assertTrue(hasattr(dm, "predict_dataset"))

        dm.setup(stage="fit")
        self.assertTrue(hasattr(dm, "full_dataset"))

        dm.setup(stage="test")
        self.assertTrue(hasattr(dm, "test_dataset"))

    def test_train_main_entrypoint(self):
        """Forces coverage of the 'if __name__ == "__main__"' block at the bottom
        of train.py.
        Note: clutters the printouts.
        """
        import runpy

        # Mock main so it doesn't actually run a full training session
        with patch("training.train.main"), patch("sys.argv", ["train.py"]):
            runpy.run_path("packages/training/train.py", run_name="__main__")

    def test_model_all_steps(self):
        """Trigger validation and test steps to hit 100% coverage on the model."""
        batch = (torch.randint(0, 20, (4, 128)), torch.randint(0, 2, (4,)))

        # This hits the lines you are likely missing
        val_loss = self.model.validation_step(batch, 0)
        test_loss = self.model.test_step(batch, 0)

        self.assertIsNotNone(val_loss)
        self.assertIsNotNone(test_loss)

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
            "rope_tt_normpostl_cls",
            "rope_tt_nonormpostl_view",
            "rope_tt_nonormpostl_sum",
            "rope_tt_nonormpostl_cls",
            "rope_nt_normpostl_view",
            "rope_nt_normpostl_sum",
            "rope_nt_normpostl_cls",
            "rope_nt_nonormpostl_view",
            "rope_nt_nonormpostl_sum",
            "rope_nt_nonormpostl_cls",
            "nope_tt_normpostl_view",
            "nope_tt_normpostl_sum",
            "nope_tt_normpostl_cls",
            "nope_tt_nonormpostl_view",
            "nope_tt_nonormpostl_sum",
            "nope_tt_nonormpostl_cls",
            "nope_nt_normpostl_view",
            "nope_nt_normpostl_sum",
            "nope_nt_normpostl_cls",
            "nope_nt_nonormpostl_view",
            "nope_nt_nonormpostl_sum",
            "nope_nt_nonormpostl_cls",
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
                        self.cfg,
                        self.num_tokens,
                        self.vocab_size_out,
                        self.length_padded,
                        archspecs,
                        archspecs2,
                    )
                    new_model.log = MagicMock()
                    new_model.trainer = MagicMock()
                    loss = new_model.training_step(batch, 0)

                    self.assertIsInstance(loss, torch.Tensor)
                    self.assertFalse(torch.isnan(loss))

    def test_all_defined_archspecs(self):
        """Dynamic coverage for every spec defined in the model file."""
        from arc.TransformerX import archspecs, archspecs2

        batch = (
            torch.randint(0, self.num_tokens, (2, self.length_padded)),
            torch.randint(0, self.vocab_size_out, (2,)),
        )

        for spec1 in archspecs.keys():
            for spec2 in archspecs2.keys():
                with self.subTest(spec1=spec1, spec2=spec2):
                    self.cfg.hyp.archspec = spec1
                    self.cfg.hyp.archspec2 = spec2
                    # Test both Compressed and Non-Compressed modes
                    for lcom_val in [None, 30]:
                        # print(spec1,spec2,lcom_val)

                        self.cfg.hyp.lcom = lcom_val

                        # Ensure dim1 == dim for transformer modes to avoid sys.exit
                        self.cfg.hyp.dim1 = self.cfg.hyp.dim

                        model = TransformerX(
                            self.cfg,
                            self.num_tokens,
                            self.vocab_size_out,
                            self.length_padded,
                            archspecs,
                            archspecs2,
                        )
                        logits = model(batch[0])
                        self.assertEqual(logits.shape, (2, self.vocab_size_out))

    def test_invalid_depth_and_layers(self):
        print("\n---test_invalid_depth_and_layers---")
        with initialize(version_base=None, config_path="../../tests/unit/data"):
            cfg = compose(
                config_name="config_test_2",
                overrides=["hyp.depth=0", "+hyp.layer_method=torch_transformer"],
            )
            # 3. Assert that the ValueError is raised during instantiation
            with self.assertRaises(ValueError) as cm:
                model = TransformerX(
                    cfg,
                    self.num_tokens,
                    self.vocab_size_out,
                    self.length_padded,
                    archspecs,
                    archspecs2,
                )
            # # 4. Verify the error message (OUTSIDE the 'with' block)
            # # We use a partial match so the test doesn't break if you tweak wording
            # self.assertIn("self.layer_method is not None", str(cm.exception)) #works

    def test_invalid_dim_and_dim1(self):
        print("\n---test_invalid_dim_and_dim1---")
        with initialize(version_base=None, config_path="../../tests/unit/data"):
            cfg = compose(
                config_name="config_test_2",
                overrides=["hyp.dim=8", "hyp.dim1=16", "+hyp.layer_method=torch_transformer"],
            )
            # 3. Assert that the ValueError is raised during instantiation
            with self.assertRaises(ValueError):
                model = TransformerX(
                    cfg,
                    self.num_tokens,
                    self.vocab_size_out,
                    self.length_padded,
                    archspecs,
                    archspecs2,
                )

    def test_invalid_archspec(self):
        print("\n---test_invalid_archspec---")
        # re-define the dictionaries. The single archspec2 is valid while the archspec are all invalid.
        archspecs = {
            "valid1": {
                "positional_encoding": "invalid",
                "layer_method": "torch_transformer",
                "norm_post_layers": False,
                "aggr_attention_method": "view",
            },
            "valid2": {
                "positional_encoding": "rope",
                "layer_method": "invalid",
                "norm_post_layers": False,
                "aggr_attention_method": "view",
            },
            "valid3": {
                "positional_encoding": "rope",
                "layer_method": "torch_transformer",
                "norm_post_layers": "invalid",
                "aggr_attention_method": "view",
            },
            "valid4": {
                "positional_encoding": "rope",
                "layer_method": "torch_transformer",
                "norm_post_layers": False,
                "aggr_attention_method": "invalid",
            },
        }
        archspecs2 = {
            "invalid_call": {"call": "L4", "n": "bn"},
        }
        batch = (
            torch.randint(0, self.num_tokens, (2, self.length_padded)),
            torch.randint(0, self.vocab_size_out, (2,)),
        )

        # pass each key combination if archspecs1 - all archspecs1 should
        # lead to failure by a sub-key
        for spec1 in archspecs.keys():
            for spec2 in archspecs2.keys():
                with self.subTest(spec1=spec1, spec2=spec2):
                    # print(f"-- spec1 {spec1} spec2 {spec2}")
                    self.cfg.hyp.archspec = spec1
                    self.cfg.hyp.archspec2 = spec2
                    # Test Non-Compressed mode
                    self.cfg.hyp.lcom = None

                    # Ensure dim1 == dim for transformer modes to avoid sys.exit
                    self.cfg.hyp.dim1 = self.cfg.hyp.dim
                    # print(f"spec1 {spec1} spec2 {spec2}")
                    # with self.assertRaises(ValueError) as cm:
                    with self.assertRaises(ValueError):
                        model = TransformerX(
                            self.cfg,
                            self.num_tokens,
                            self.vocab_size_out,
                            self.length_padded,
                            archspecs,
                            archspecs2,
                        )

    def test_invalid_archspec2(self):
        print("\n---test_invalid_archspec2---")
        # re-define the dictionaries. The single archspec is valid while all the archspec2 are invalid.
        archspecs = {
            "valid": {
                "positional_encoding": "rope",
                "layer_method": "torch_transformer",
                "norm_post_layers": False,
                "aggr_attention_method": "view",
            }
        }
        archspecs2 = {
            "invalid_call": {"call": "invalid", "n": "bn"},
            "invalid_n": {"call": "L4", "n": "invalid"},
        }
        batch = (
            torch.randint(0, self.num_tokens, (2, self.length_padded)),
            torch.randint(0, self.vocab_size_out, (2,)),
        )
        # pass each key combination if archspecs2 - all archspecs2 should
        # lead to failure by a sub-key
        for spec1 in archspecs.keys():
            for spec2 in archspecs2.keys():
                with self.subTest(spec1=spec1, spec2=spec2):
                    self.cfg.hyp.archspec = spec1
                    self.cfg.hyp.archspec2 = spec2
                    # Test Non-Compressed mode
                    self.cfg.hyp.lcom = None

                    # Ensure dim1 == dim for transformer modes to avoid sys.exit
                    self.cfg.hyp.dim1 = self.cfg.hyp.dim
                    # with self.assertRaises(ValueError) as cm:
                    with self.assertRaises(ValueError):
                        model = TransformerX(
                            self.cfg,
                            self.num_tokens,
                            self.vocab_size_out,
                            self.length_padded,
                            archspecs,
                            archspecs2,
                        )


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
