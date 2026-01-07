# tests/utils/test_config_loader.py
"""
To get the correct installed packages
python -m unittest tests.test_config_loader
PYTHONPATH=./packages pytest tests/
"""

import sys
import unittest

import torch
from omegaconf import OmegaConf
from training.arc.TransformerX import TransformerX

# from training.dataloaders.dataloaders import BinaryPairkeyDataModule


class TestTransformerX(unittest.TestCase):
    def setUp(self):
        """Set up a mock configuration similar to your config_test_2.yaml"""
        self.num_tokens = 20
        self.vocab_size_out = 2
        self.length_padded = 128

        self.cfg = OmegaConf.create(
            {
                # "seed"  : 42,
                # "verboselevel" : "ERROR",
                # "debug_mode" : False,
                # "train_features_path" : "../../bp/AlphaX/proc/sim_data/Tests/x_Test_50_50_TOK_test_AT_cond2_aligned_test_binary_pairkey.h5",
                # "train_labels_path" : "../../bp/AlphaX/proc/sim_data/Tests/cl_Test_50_50_TOK_test_AT_cond2_aligned_test_binary_pairkey.h5",
                # "debug_features_path" : "",
                # "debug_labels_path" : "",
                # "pair_indices_name" : "pair_indices",
                # "dataset_name_host" : "host_vec_1",
                # "dataset_name_phage" : "phage_vec_1",
                # "reader_mode" : "binary_pairkey",
                # "architecture" : "TransformerX",
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
                    self.model = TransformerX(
                        self.cfg, self.num_tokens, self.vocab_size_out, self.length_padded
                    )
                    loss = self.model.training_step(batch, 0)

                    self.assertIsInstance(loss, torch.Tensor)
                    self.assertFalse(torch.isnan(loss))


if __name__ == "__main__":
    unittest.main(exit=False)
