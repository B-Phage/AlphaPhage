import unittest
from unittest.mock import MagicMock

import torch
import torch.nn as nn
from arc.transformer_archspecs import archspec2_list, archspec_list, archspecs, archspecs2
from arc.transformer_module import TransformerX
from omegaconf import OmegaConf


class TestTransformerChecks(unittest.TestCase):

    def setUp(self):
        """Build a base valid config that we can break in each test."""
        self.arch_key = "valid_arch"
        self.archspecs = {
            self.arch_key: {
                "positional_encoding": "rope",
                "layer_method": "torch_transformer",
                "norm_post_layers": False,
                "aggr_attention_method": "view",
            }
        }
        self.archspecs2 = {"L4_bn": {"call": "L4", "n": "bn"}}
        self.cfg = OmegaConf.create(
            {
                "hyp": {
                    "archspec": self.arch_key,
                    "archspec2": "L4_bn",
                    "dim": 8,
                    "dim1": 8,
                    "dim2": 8,
                    "heads": 4,
                    "mult": 4,
                    "depth": 1,
                    "lcom": None,
                },
                "optimizer": {"lr": 0.001},
            }
        )
        # Standard init arguments
        self.args = [100, 10, 50, self.archspecs, self.archspecs2]

    def test_invalid_positional_encoding(self):
        """Triggers error if positional_encoding is not 'rope' or None."""
        self.archspecs[self.arch_key]["positional_encoding"] = "fourier"
        with self.assertRaisesRegex(ValueError, "positional_encoding 'fourier' not recognised"):
            TransformerX(self.cfg, *self.args)

    def test_invalid_layer_method(self):
        """Triggers error if layer_method is not 'torch_transformer' or None."""
        self.archspecs[self.arch_key]["layer_method"] = "custom_transformer"
        with self.assertRaisesRegex(ValueError, "layer_method 'custom_transformer' not recognised"):
            TransformerX(self.cfg, *self.args)

    def test_invalid_norm_post_layers(self):
        """Triggers error if norm_post_layers is not a boolean."""
        self.archspecs[self.arch_key]["norm_post_layers"] = "maybe"
        with self.assertRaisesRegex(ValueError, "norm_post_layers 'maybe' not recognised"):
            TransformerX(self.cfg, *self.args)

    def test_invalid_aggr_method(self):
        """Triggers error if aggr_attention_method is invalid."""
        self.archspecs[self.arch_key]["aggr_attention_method"] = "max_pool"
        with self.assertRaisesRegex(ValueError, "aggr_attention_method 'max_pool' not recognised"):
            TransformerX(self.cfg, *self.args)

    def test_invalid_norm_type(self):
        """Triggers error if the MLP norm 'n' is not bn, ln, or nn."""
        self.archspecs2["L4_bn"]["n"] = "group_norm"
        with self.assertRaisesRegex(ValueError, "self.n 'group_norm' not recognised"):
            TransformerX(self.cfg, *self.args)

    def test_depth_zero_conflict(self):
        """Ensures depth > 0 if a layer_method is requested."""
        self.cfg.hyp.depth = 0
        self.archspecs[self.arch_key]["layer_method"] = "torch_transformer"
        with self.assertRaisesRegex(
            ValueError, "self.depth == 0 and self.layer_method is not None"
        ):
            TransformerX(self.cfg, *self.args)

    def test_dim_mismatch_check(self):
        """Ensures dim1 == dim when using the transformer encoder."""
        self.cfg.hyp.dim = 8
        self.cfg.hyp.dim1 = 16
        self.archspecs[self.arch_key]["layer_method"] = "torch_transformer"
        with self.assertRaisesRegex(
            ValueError, "self.layer_method is not None and self.dim1 != self.dim"
        ):
            TransformerX(self.cfg, *self.args)


class TestTransformerX(unittest.TestCase):

    def setUp(self):
        """Standard unittest setup: creates a fresh mock config for every test."""
        self.archspecs = {
            "rope_tt_normpostl_view": {
                "positional_encoding": "rope",
                "layer_method": "torch_transformer",
                "norm_post_layers": False,
                "aggr_attention_method": "view",
            }
        }
        self.archspecs2 = {
            "L4_bn": {"call": "L4", "n": "bn"},
            "L3_bn": {"call": "L3", "n": "bn"},  # Added for L3 test
        }
        self.cfg = OmegaConf.create(
            {
                "hyp": {
                    "archspec": "rope_tt_normpostl_view",
                    "archspec2": "L4_bn",
                    "dim": 8,
                    "lcom": None,  # Set to None to simplify initial test
                    "dim1": 8,
                    "dim2": 8,
                    "heads": 4,
                    "mult": 4,
                    "depth": 1,
                    "vocab_size_out": 2,
                },
                "optimizer": {"lr": 0.001},  # Required for configure_optimizers if called
            }
        )
        self.num_tokens = 100
        self.vocab_size_out = 10
        self.length_padded = 50

    def test_output_shape(self):
        """Verifies the output matches the expected Batch x VocabOut shape."""
        batch_size = 4
        model = TransformerX(
            self.cfg,
            self.num_tokens,
            self.vocab_size_out,
            self.length_padded,
            self.archspecs,
            self.archspecs2,
        )

        x = torch.randint(0, self.num_tokens, (batch_size, self.length_padded))
        output = model(x)

        self.assertEqual(output.shape, (batch_size, self.vocab_size_out))

    def test_layer_counts_l3(self):
        """Ensures L3 logic creates exactly 3 linear layers and 2 norms."""
        # Update config to L3
        self.cfg.hyp.archspec2 = "L3_bn"
        model = TransformerX(
            self.cfg,
            self.num_tokens,
            self.vocab_size_out,
            self.length_padded,
            self.archspecs,
            self.archspecs2,
        )

        # NOTE: We must check model.model because the linears live in the Classifier
        self.assertEqual(len(model.model.linears), 3)
        self.assertEqual(len(model.model.norms), 2)
        self.assertEqual(model.model.linears[-1].out_features, self.vocab_size_out)

    def test_gradient_flow(self):
        """MLOps Critical: Ensures no layers are detached from the computational graph."""
        batch_size = 2  # Fixed: Define batch_size locally
        model = TransformerX(
            self.cfg,
            self.num_tokens,
            self.vocab_size_out,
            self.length_padded,
            self.archspecs,
            self.archspecs2,
        )

        x = torch.randint(0, self.num_tokens, (batch_size, self.length_padded))

        output = model(x)
        loss = output.sum()
        loss.backward()

        # Check model.model.linears
        self.assertIsNotNone(model.model.linears[0].weight.grad)
        self.assertGreater(model.model.linears[0].weight.grad.abs().sum().item(), 0)

    def test_training_step_all_arcspecs(self):
        """Test if training_step computes a loss for all archspec settings."""
        batch = (
            torch.randint(0, self.num_tokens, (4, self.length_padded)),
            torch.randint(0, self.vocab_size_out, (4,)),
        )

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


if __name__ == "__main__":
    unittest.main()
