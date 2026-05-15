import os
import shutil
import unittest
import zipfile

import yaml
from utils.zip_db import (
    add_file_to_zip,
    add_folder_to_zip,
    create_mapped_zip,
    load_config,
    process_mapping,
)


class TestZipUtility(unittest.TestCase):
    """Unit tests for the ZIP mapping utility."""

    def setUp(self):
        """Create a temporary directory structure for testing."""
        self.test_dir = "test_root"
        self.db_dir = os.path.join(self.test_dir, "databases/hmm/v4/phrog")
        self.target_zip = "test_output.zip"
        self.config_file = "test_config.yaml"

        os.makedirs(self.db_dir, exist_ok=True)

        # Create dummy files
        self.file1 = os.path.join(self.db_dir, "data.txt")
        with open(self.file1, "w") as f:
            f.write("test data")

        self.standalone_file = os.path.join(self.test_dir, "standalone.log")
        with open(self.standalone_file, "w") as f:
            f.write("log content")

    def tearDown(self):
        """Clean up temporary files and directories."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        if os.path.exists(self.target_zip):
            os.remove(self.target_zip)
        if os.path.exists(self.config_file):
            os.remove(self.config_file)

    def test_load_config(self):
        """Test that YAML is loaded correctly into a dictionary."""
        config_data = {"target_zip": "out.zip", "mappings": [{"a": "b"}]}
        with open(self.config_file, "w") as f:
            yaml.dump(config_data, f)

        result = load_config(self.config_file)
        self.assertEqual(result["target_zip"], "out.zip")
        self.assertEqual(result["mappings"][0]["a"], "b")

    def test_add_file_to_zip(self):
        """Test adding a single file to a zip archive."""
        with zipfile.ZipFile(self.target_zip, "w") as zf:
            add_file_to_zip(zf, self.file1, "archived_data.txt")

        with zipfile.ZipFile(self.target_zip, "r") as zf:
            self.assertIn("archived_data.txt", zf.namelist())

    def test_add_folder_to_zip(self):
        """Test recursive folder addition."""
        with zipfile.ZipFile(self.target_zip, "w") as zf:
            add_folder_to_zip(zf, self.db_dir, "db_backup")

        with zipfile.ZipFile(self.target_zip, "r") as zf:
            # Check if the relative file path exists inside the ZIP
            self.assertIn("db_backup/data.txt", zf.namelist())

    def test_process_mapping_missing_path(self):
        """Test behavior when a source path does not exist."""
        with zipfile.ZipFile(self.target_zip, "w") as zf:
            # Should print warning and return instead of crashing
            process_mapping(zf, "non_existent_path", "target")

        with zipfile.ZipFile(self.target_zip, "r") as zf:
            self.assertEqual(len(zf.namelist()), 0)

    def test_process_mapping_file(self):
        """Test process_mapping identifying a file."""
        with zipfile.ZipFile(self.target_zip, "w") as zf:
            process_mapping(zf, self.standalone_file, "logs/init.log")

        with zipfile.ZipFile(self.target_zip, "r") as zf:
            self.assertIn("logs/init.log", zf.namelist())

    def test_process_mapping_directory(self):
        """Test process_mapping identifying a directory."""
        with zipfile.ZipFile(self.target_zip, "w") as zf:
            process_mapping(zf, self.db_dir, "databases/phrog")

        with zipfile.ZipFile(self.target_zip, "r") as zf:
            self.assertIn("databases/phrog/data.txt", zf.namelist())

    def test_create_mapped_zip_full_flow(self):
        """Orchestration test ensuring the entire flow works from config to ZIP."""
        config_content = {
            "target_zip": self.target_zip,
            "mappings": [
                {self.db_dir: "databases/target_path"},
                {self.standalone_file: "single_file.log"},
            ],
        }
        with open(self.config_file, "w") as f:
            yaml.dump(config_content, f)

        create_mapped_zip(self.config_file)

        self.assertTrue(os.path.exists(self.target_zip))
        with zipfile.ZipFile(self.target_zip, "r") as zf:
            names = zf.namelist()
            self.assertIn("databases/target_path/data.txt", names)
            self.assertIn("single_file.log", names)


if __name__ == "__main__":
    unittest.main()
