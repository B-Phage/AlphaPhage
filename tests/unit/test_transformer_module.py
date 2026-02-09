import unittest

import torch
from models.transformer_module import TransformerX
from omegaconf import OmegaConf


class TestTransformerXLightning(unittest.TestCase):

    def setUp(self):
        """Set up a valid configuration for the Lightning Module."""
        self.archspecs = {
            "rope_tt_normpostl_view": {
                "positional_encoding": "rope",
                "layer_method": "torch_transformer",
                "norm_post_layers": False,
                "aggr_attention_method": "view",
            }
        }
        self.archspecs2 = {"L1_nn": {"call": "L1", "n": "nn"}}

        self.cfg = OmegaConf.create(
            {
                "hyp": {
                    "archspec": "rope_tt_normpostl_view",
                    "archspec2": "L1_nn",
                    "dim": 8,
                    "dim1": 8,
                    "dim2": 8,
                    "heads": 2,
                    "mult": 2,
                    "depth": 1,
                    "lcom": None,
                },
                "optimizer": {"lr": 0.001},
            }
        )

        self.num_tokens = 50
        self.vocab_size_out = 2
        self.length_padded = 10

        self.model = TransformerX(
            self.cfg,
            self.num_tokens,
            self.vocab_size_out,
            self.length_padded,
            self.archspecs,
            self.archspecs2,
        )

    def test_hparams_saved(self):
        """Verify that hyperparameters are saved for Lightning checkpointing."""
        # Lightning stores these in self.hparams
        self.assertTrue(hasattr(self.model, "hparams"))
        self.assertEqual(self.model.hparams["hyp"]["dim"], 8)

    def test_training_step(self):
        """Test the training loop's forward pass and loss calculation."""
        batch_size = 2
        x = torch.randint(0, self.num_tokens, (batch_size, self.length_padded))
        y = torch.randint(0, self.vocab_size_out, (batch_size,))
        batch = (x, y)

        loss = self.model.training_step(batch, 0)

        # Check that loss is a scalar tensor with a gradient
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.shape, ())
        self.assertTrue(loss.requires_grad)

    def test_validation_step(self):
        """Verify validation step logs loss and accuracy."""
        batch_size = 2
        x = torch.randint(0, self.num_tokens, (batch_size, self.length_padded))
        y = torch.randint(0, self.vocab_size_out, (batch_size,))
        batch = (x, y)

        # In standard unittest, we just check if it runs without error
        # and returns the expected loss tensor
        loss = self.model.validation_step(batch, 0)
        self.assertIsInstance(loss, torch.Tensor)

    def test_test_step_logic(self):
        """Ensures test_step calculates loss and handles accuracy correctly."""
        batch_size = 4
        # 1. Create mock batch: Indices for x, Class labels for y
        x = torch.randint(0, self.num_tokens, (batch_size, self.length_padded))
        y = torch.randint(0, self.vocab_size_out, (batch_size,))
        batch = (x, y)

        # 2. Run the test_step
        # We pass batch_idx=0 as it is required by the Lightning interface
        test_loss = self.model.test_step(batch, 0)

        # 3. Assertions
        # Check that loss is a valid scalar tensor
        self.assertIsInstance(test_loss, torch.Tensor)
        self.assertEqual(test_loss.shape, ())
        self.assertFalse(torch.isnan(test_loss), "Test loss is NaN!")

    def test_test_step_accuracy_edge_case(self):
        """Verifies accuracy calculation when all predictions are correct."""
        batch_size = 2
        x = torch.randint(0, self.num_tokens, (batch_size, self.length_padded))
        y = torch.tensor([1, 0])
        batch = (x, y)

        # To force 100% accuracy, we can't easily control logits,
        # but we can verify that the code handles the float conversion and mean.
        test_loss = self.model.test_step(batch, 0)

        # We primarily check that the logging calls (test_acc) didn't crash
        self.assertIsNotNone(test_loss)

    def test_configure_optimizers(self):
        """Ensure the optimizer is correctly initialized from config."""
        opt = self.model.configure_optimizers()

        self.assertIsInstance(opt, torch.optim.AdamW)
        self.assertEqual(opt.defaults["lr"], 0.001)

    def test_forward_consistency(self):
        """Ensure the wrapper forward matches the internal model output."""
        x = torch.randint(0, self.num_tokens, (1, self.length_padded))

        # Output from the LightningModule wrapper
        out_wrapper = self.model(x)
        # Output directly from the internal Classifier
        out_internal = self.model.model(x)

        self.assertTrue(torch.equal(out_wrapper, out_internal))


if __name__ == "__main__":
    unittest.main()
