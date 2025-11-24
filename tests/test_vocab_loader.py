# tests/utils/test_vocab_loader.py
"""
To get the correct installed packages
python -m unittest tests.test_vocab_loader
PYTHONPATH=./packages pytest tests/
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from omegaconf import DictConfig, OmegaConf


class TestVocabLoader(unittest.TestCase):

    def setUp(self):
        # Create a temporary test directory
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

        # --- Create temporary vocab files ---
        self.vocab_file = self.temp_path / "vocab.yaml"
        self.test_vocab_file = self.temp_path / "test_vocab.yaml"

        vocab_content = OmegaConf.create(
            {
                "token_vocab": {"A": 1, "B": 2},
                "label_vocab": {"pos": 1, "neg": 0},
            }
        )
        OmegaConf.save(vocab_content, self.vocab_file)

        test_vocab_content = OmegaConf.create(
            {
                "token_vocab": {"X": 10, "Y": 20},
            }
        )
        OmegaConf.save(test_vocab_content, self.test_vocab_file)

        # --- Create temporary params.yaml ---
        self.params_file = self.temp_path / "params.yaml"
        params_content = OmegaConf.create(
            {
                "vocab": {
                    "file": str(self.vocab_file),
                    "test_file": str(self.test_vocab_file),
                }
            }
        )
        OmegaConf.save(params_content, self.params_file)

        # Patch the PARAMS_PATH inside vocab_loader so it uses our temp copy
        patcher = patch("utils.vocab_loader.PARAMS_PATH", self.params_file)
        self.addCleanup(patcher.stop)
        self.mock_params_path = patcher.start()

        # Import *after* mocking to ensure loader picks up test paths
        global vocab_loader
        from utils import vocab_loader

        self.loader = vocab_loader

        # Clear all LRU caches so each test starts clean
        # self.loader._load_params.cache_clear()
        # self.loader._load_vocab_file.cache_clear()

    def tearDown(self):
        self.temp_dir.cleanup()

    # --------------- TESTS ----------------

    def test_normal_loading(self):
        vocab = self.loader.get_vocab()
        self.assertIn("token_vocab", vocab)
        self.assertEqual(vocab.token_vocab.A, 1)
        self.assertEqual(vocab.label_vocab.pos, 1)

    def test_test_mode_loading(self):
        vocab = self.loader.get_vocab(test_mode=True)
        self.assertIn("token_vocab", vocab)
        self.assertEqual(vocab.token_vocab.X, 10)
        self.assertNotIn("label_vocab", vocab)

    def test_override_path(self):
        override_vocab = OmegaConf.create({"override": {"v": 123}})
        override_file = self.temp_path / "override.yaml"
        OmegaConf.save(override_vocab, override_file)

        vocab = self.loader.get_vocab(override_path=str(override_file))
        self.assertEqual(vocab.override.v, 123)

    def test_multiple_vocab_sections(self):
        vocab = self.loader.get_vocab()
        print(type(vocab.token_vocab))
        self.assertIsInstance(vocab.token_vocab, DictConfig)
        self.assertIsInstance(vocab.label_vocab, DictConfig)
        self.assertEqual(len(vocab.token_vocab), 2)
        self.assertEqual(len(vocab.label_vocab), 2)

    # def test_multiple_vocab_sections_dict(self):
    #     # testing future replacement option
    #     vocab = self.loader._load_vocab_file_as_dict(self.params_file)
    #     print(type(vocab['token_vocab']))
    #     self.assertIsInstance(vocab['token_vocab'], dict)
    #     self.assertIsInstance(vocab['label_vocab'], dict)
    #     self.assertEqual(len(vocab['token_vocab']), 2)
    #     self.assertEqual(len(vocab['label_vocab']), 2)

    # def test_caching_behavior(self):
    #     """Ensure LRU caching prevents multiple loads."""
    #     # internal counters
    #     original_load_params = self.loader._load_params
    #     original_load_vocab = self.loader._load_vocab_file

    #     with patch.object(self.loader, "_load_params", wraps=original_load_params) as mock_params, \
    #          patch.object(self.loader, "_load_vocab_file", wraps=original_load_vocab) as mock_vocab:

    #         # First call
    #         self.loader.get_vocab()
    #         # Second call (should use cache)
    #         self.loader.get_vocab()

    #     # _load_params called once
    #     mock_params.assert_called_once()
    #     # _load_vocab_file called once for the main vocab file
    #     mock_vocab.assert_called_once()

    def test_missing_test_file(self):
        """Test mode should fail if no test_file configured."""
        # Rewrite params without test_file
        params_no_test = OmegaConf.create({"vocab": {"file": str(self.vocab_file)}})
        OmegaConf.save(params_no_test, self.params_file)
        # self.loader._load_params.cache_clear()

        with self.assertRaises(ValueError):
            self.loader.get_vocab(test_mode=True)


if __name__ == "__main__":
    unittest.main(exit=False)
