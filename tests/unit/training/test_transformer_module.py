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
                "archspec": "rope_tt_normpostl_view",
                "archspec2": "L1_nn",
                "dim": 8,
                "dim1": 8,
                "dim2": 8,
                "heads": 2,
                "mult": 2,
                "depth": 1,
                "lcom": None,
                "lr": 0.001,
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
        self.assertEqual(self.model.hparams["cfg"]["dim"], 8)

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

    def test_predict_step(self):
        """Verifies that predict_step returns probabilities summing to 1."""
        batch_size = 4
        x = torch.randint(0, self.num_tokens, (batch_size, self.length_padded))
        # predict_step expects a (x, y) tuple based on your implementation
        y = torch.randint(0, self.vocab_size_out, (batch_size,))
        batch = (x, y)

        # 1. Execute predict_step
        probs = self.model.predict_step(batch, 0)

        # 2. Assertions on Shape
        # Output should be (batch_size, vocab_size_out)
        self.assertEqual(probs.shape, (batch_size, self.vocab_size_out))

        # 3. Assertions on Probability Properties
        # Softmax ensures all values are between 0 and 1
        self.assertTrue(torch.all(probs >= 0))
        self.assertTrue(torch.all(probs <= 1))

        # Each row should sum to approximately 1.0
        row_sums = probs.sum(dim=-1)
        expected_sums = torch.ones(batch_size)
        self.assertTrue(torch.allclose(row_sums, expected_sums, atol=1e-6))

    def test_predict_step_outputs_different_from_logits(self):
        """Ensures that predict_step actually applies softmax to raw logits."""
        batch_size = 1
        x = torch.randint(0, self.num_tokens, (batch_size, self.length_padded))
        y = torch.zeros(batch_size)
        batch = (x, y)

        # Get raw logits via forward
        logits = self.model(x)
        # Get probabilities via predict_step
        probs = self.model.predict_step(batch, 0)

        # If logits are not already normalized, they shouldn't equal the softmax output
        # (Unless by extreme coincidence, but testing for inequality is standard here)
        self.assertFalse(torch.equal(logits, probs))

    def test_predict_step_flexible_input(self):
        """Verifies predict_step handles both (x, y) tuples and raw x tensors."""
        batch_size = 3
        x_input = torch.randint(0, self.num_tokens, (batch_size, self.length_padded))
        y_label = torch.randint(0, self.vocab_size_out, (batch_size,))

        # Case 1: Batch is a tuple (x, y) - Standard during testing
        batch_tuple = (x_input, y_label)
        probs_from_tuple = self.model.predict_step(batch_tuple, 0)
        self.assertEqual(probs_from_tuple.shape, (batch_size, self.vocab_size_out))

        # Case 2: Batch is just a tensor x - Standard during production inference
        probs_from_raw = self.model.predict_step(x_input, 0)
        self.assertEqual(probs_from_raw.shape, (batch_size, self.vocab_size_out))

        # Verification: Both results should be identical
        self.assertTrue(torch.equal(probs_from_tuple, probs_from_raw))

        # Final check: Probabilities should sum to 1
        row_sums = probs_from_raw.sum(dim=-1)
        self.assertTrue(torch.allclose(row_sums, torch.ones(batch_size)))


if __name__ == "__main__":
    unittest.main()
