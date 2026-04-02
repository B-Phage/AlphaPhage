# tests/utils/test_config_loader.py
"""
To get the correct installed packages
python -m unittest tests.test_config_loader
PYTHONPATH=./packages pytest tests/
"""

import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from omegaconf import DictConfig, OmegaConf
from utils.config_loader import _load_vocab_file, get_config_generic


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

        # Patch the PARAMS_PATH inside config_loader so it uses our temp copy
        patcher = patch("utils.config_loader.PARAMS_PATH", self.params_file)
        self.addCleanup(patcher.stop)
        self.mock_params_path = patcher.start()

        # Import *after* mocking to ensure loader picks up test paths
        global config_loader
        from utils import config_loader

        self.loader = config_loader

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
        self.assertIsInstance(vocab.token_vocab, DictConfig)
        self.assertIsInstance(vocab.label_vocab, DictConfig)
        self.assertEqual(len(vocab.token_vocab), 2)
        self.assertEqual(len(vocab.label_vocab), 2)

    def test_missing_test_file(self):
        """Test mode should fail if no test_file configured."""
        # Rewrite params without test_file
        params_no_test = OmegaConf.create({"vocab": {"file": str(self.vocab_file)}})
        OmegaConf.save(params_no_test, self.params_file)

        with self.assertRaises(ValueError):
            self.loader.get_vocab(test_mode=True)

    def test_missing_file_raises_filenotfounderror(self):
        """
        Tests that calling _load_vocab_file with a non-existent path
        correctly raises a FileNotFoundError.
        """
        # 1. Define the path of a file that certainly does not exist
        missing_path = "/tmp/file/does/not/exist_123456789.yaml"

        # Use a context manager to assert that the specific exception is raised
        # We expect FileNotFoundError to be raised when called with the missing path.
        with self.assertRaises(FileNotFoundError) as cm:
            self.loader._load_vocab_file(missing_path)

        # Verify the exception message/argument
        # The exception object should contain the path that failed.
        # Note: The FileNotFoundError in Python 3 accepts the path object itself
        # as its argument, which is then often represented as a string in the message.

        # Check the first argument passed to the exception constructor
        raised_path = cm.exception.args[0]

        # We need to compare the path used for the test with the path in the exception.
        self.assertEqual(
            str(raised_path),
            missing_path,
            "The FileNotFoundError should contain the path of the missing file.",
        )

    def test_prodigal_config_loading(self):
        conf = self.loader.get_config(
            override_path="ml_assets/config/preprocessing_ecoli_v1.0.0.yaml",
            params_set="hmm_config",
        )
        gen_conf = self.loader.get_config(
            override_path="ml_assets/config/preprocessing_ecoli_v1.0.0.yaml",
            params_set="general",
        )
        self.assertEqual(gen_conf.name, "bacteria")
        self.assertEqual(conf.hmm_scan_method, "pfam")

        with self.assertRaises(FileNotFoundError):
            conf = self.loader.get_config("missing.yaml")

        with self.assertRaises(KeyError):
            conf = self.loader.get_config("params.yaml")


class TestConfigLoaderGeneric(unittest.TestCase):

    def setUp(self):
        # Sample data structures for mocking
        self.basic_config = {"level1": {"level2": "value_final"}, "simple": "direct"}
        self.secondary_config = {"nested": "found_it"}
        # Create a temporary directory for YAML files
        self.test_dir_path = tempfile.mkdtemp()
        self.test_dir = Path(self.test_dir_path)

    def tearDown(self):
        # Remove the directory after the test
        shutil.rmtree(self.test_dir_path)

    @patch("utils.config_loader.OmegaConf.load")
    @patch("utils.config_loader.Path.exists")
    def test_get_config_generic_no_params_set(self, mock_exists, mock_load):
        """Should return the entire config when params_set is None."""
        mock_exists.return_value = True
        mock_load.return_value = OmegaConf.create(self.basic_config)

        result = get_config_generic("dummy.yaml")
        self.assertEqual(result.simple, "direct")
        self.assertIsInstance(result, DictConfig)

    @patch("packages.utils.config_loader.OmegaConf.load")
    def test_get_config_generic_list_traversal(self, mock_load):
        """Should traverse a list of keys correctly."""
        mock_load.return_value = OmegaConf.create(self.basic_config)

        # Testing list traversal: ["level1", "level2"]
        result = get_config_generic("dummy.yaml", params_set=["level1", "level2"])
        self.assertEqual(result, "value_final")

    @patch("packages.utils.config_loader.OmegaConf.load")
    def test_get_config_generic_two_step_traversal(self, mock_load):
        """Should load a second config file if params1 is a string path."""
        # Setup: First call returns a string path, Second call returns a dict
        mock_load.side_effect = [
            OmegaConf.create({"config_path": "second.yaml"}),  # First file
            OmegaConf.create(self.secondary_config),  # Second file
        ]

        result = get_config_generic(
            path="first.yaml", params_set="config_path", params_set_2="nested"
        )

        self.assertEqual(result, "found_it")
        self.assertEqual(mock_load.call_count, 2)

    @patch("packages.utils.config_loader.OmegaConf.load")
    def test_get_config_generic_key_error(self, mock_load):
        """Should raise KeyError if a section is missing."""
        mock_load.return_value = OmegaConf.create(self.basic_config)

        with self.assertRaises(KeyError):
            get_config_generic("dummy.yaml", params_set="non_existent")

    def test_get_config_generic_errors_and_lists(self):
        # Setup temporary files
        list_yaml = self.test_dir / "list.yaml"
        OmegaConf.save([1, 2, 3], list_yaml)

        nested_yaml = self.test_dir / "nested.yaml"
        OmegaConf.save({"a": {"b": "target"}}, nested_yaml)

        # 1. Trigger Line 92: First file is a list
        with self.assertRaises(TypeError):
            get_config_generic(str(list_yaml))

        # 2. Trigger Line 101: Key missing in a list traversal (params_set)
        with self.assertRaises(KeyError):
            get_config_generic(str(nested_yaml), params_set=["a", "nonexistent"])

        # 3. Trigger Line 125: Second file is a list
        # Create a file that points to the list file
        pointer_yaml = self.test_dir / "pointer.yaml"
        OmegaConf.save({"next_file": str(list_yaml)}, pointer_yaml)

        with self.assertRaises(TypeError):
            get_config_generic(str(pointer_yaml), params_set="next_file", params_set_2="item")

    def test_get_config_generic_params_set_2_list_and_errors(self):
        # Create a chain of configs
        final_yaml = self.test_dir / "final.yaml"
        OmegaConf.save({"level1": {"level2": "found_me"}}, final_yaml)

        start_yaml = self.test_dir / "start.yaml"
        OmegaConf.save({"path_to_final": str(final_yaml)}, start_yaml)

        # 4. Trigger Lines 130-135: params_set_2 is a LIST
        # Navigates start.yaml -> path_to_final -> level1 -> level2
        result = get_config_generic(
            str(start_yaml), params_set="path_to_final", params_set_2=["level1", "level2"]
        )
        self.assertEqual(result, "found_me")

        # 5. Trigger Line 139: params_set_2 (string) key missing in second file
        with self.assertRaises(KeyError):
            get_config_generic(
                str(start_yaml), params_set="path_to_final", params_set_2="wrong_key"
            )

    def test_get_config_generic_type_error_params2(self):
        """Trigger the final else
        # params_set_2 is provided, but params1 is NOT a string (path)
        """
        bad_chain = self.test_dir / "bad_chain.yaml"
        OmegaConf.save({"not_a_path": 123}, bad_chain)

        with self.assertRaises(TypeError):
            get_config_generic(str(bad_chain), params_set="not_a_path", params_set_2="anything")

    def test_get_config_generic_key_error_in_params_set_2_list(self):
        """
        Triggers KeyError when traversing params_set_2 as a list
        in the second config file.
        """
        # 1. Create the second config file with partial data
        second_yaml = self.test_dir / "second_config.yaml"
        OmegaConf.save({"level_one": {"something_else": "data"}}, second_yaml)

        # 2. Create the primary config file pointing to the second one
        first_yaml = self.test_dir / "first_config.yaml"
        OmegaConf.save({"external_link": str(second_yaml)}, first_yaml)

        # 3. Attempt to traverse second_config.yaml using a list that contains a missing key
        # 'level_one' exists, but 'missing_key' does not.
        invalid_list_traversal = ["level_one", "missing_key"]

        with self.assertRaises(KeyError) as cm:
            get_config_generic(
                path=str(first_yaml),
                params_set="external_link",
                params_set_2=invalid_list_traversal,
            )

        # Verify the error message contains the missing key and the correct file path
        self.assertIn("missing_key section missing in config file", str(cm.exception))
        self.assertIn("second_config.yaml", str(cm.exception))


if __name__ == "__main__":
    unittest.main(exit=False)

import pytest


def test_load_vocab_file_raises_type_error_on_list(tmp_path):
    # 1. Setup: Create a YAML file that contains a list, not a dict
    vocab_file = tmp_path / "vocab_list.yaml"
    vocab_file.write_text("- item1\n- item2\n- item3")

    # 2. Act & Assert: Verify that it raises a TypeError
    # We expect the error message to mention 'ListConfig'
    with pytest.raises(TypeError, match="Expected DictConfig, got ListConfig"):
        _load_vocab_file(str(vocab_file))
