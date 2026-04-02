import json
import tempfile
import unittest
from pathlib import Path

from orchestration.orchestrator import Orchestrator


# Helpers
def dummy_worker(input_path: Path, outputs: list):
    for out in outputs:
        out.write_text(f"Processed {input_path.read_text()}")


def dummy_output_func(input_path: Path):
    return [
        input_path.parent / f"{input_path.stem}_out1.txt",
        input_path.parent / f"{input_path.stem}_out2.txt",
    ]


def ordered(obj):
    """
    Recursively sort lists within an object to allow comparison
    regardless of element order in arrays.
    """
    if isinstance(obj, dict):
        # Sort items in a dictionary by key, then recursively sort values
        return sorted((k, ordered(v)) for k, v in obj.items())
    if isinstance(obj, list):
        # Recursively sort the list elements
        return sorted(ordered(x) for x in obj)
    else:
        # Return primitives as is
        return obj


def failing_worker(input_path, outputs):
    raise RuntimeError("Deliberate failure from worker")


class TestOrchestrator(unittest.TestCase):

    # Helpers
    def load_json_file(self, file_path):
        """Helper to load JSON data from a file."""
        try:
            with open(file_path, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            self.fail(f"Test file not found: {file_path}")
        except json.JSONDecodeError:
            self.fail(f"Invalid JSON in file: {file_path}")

    def setUp(self):
        self.tempdir = Path(tempfile.mkdtemp())
        self.hash_dir = self.tempdir / "hashes"
        self.hash_dir.mkdir()

        # Create input files
        self.input1 = self.tempdir / "a.txt"
        self.input2 = self.tempdir / "b.txt"
        self.input1.write_text("AAA")
        self.input2.write_text("BBB")

    def test_orchestrator_basic_run(self):
        orch = Orchestrator(
            workflow_name="testflow",
            hash_store_dir=self.hash_dir,
            logger_name="testlogger",
        )

        orch.run(
            inputs=[self.input1, self.input2],
            output_func=dummy_output_func,
            worker=dummy_worker,
            code_id="v1",
            parallelism=2,
            force=False,
        )

        # Check outputs created
        out1 = self.tempdir / "a_out1.txt"
        out2 = self.tempdir / "a_out2.txt"
        self.assertTrue(out1.exists())
        self.assertTrue(out2.exists())

        # Check hashes recorded
        stored_hash_file = self.hash_dir / "testflow.json"
        self.assertTrue(stored_hash_file.exists())

        with open(stored_hash_file) as f:
            hashes = json.load(f)
        self.assertIn(str(self.input1), hashes)

    def test_force_rerun(self):
        orch = Orchestrator(
            workflow_name="testflow2",
            hash_store_dir=self.hash_dir,
            logger_name="testlogger2",
        )

        orch.run(
            inputs=[self.input1],
            output_func=dummy_output_func,
            worker=dummy_worker,
            code_id="v1",
        )

        # Modify nothing — forcing should still rerun
        orch.run(
            inputs=[self.input1],
            output_func=dummy_output_func,
            worker=dummy_worker,
            code_id="v1",
            force=True,
        )

    def test_change_rerun(self):
        orch = Orchestrator(
            workflow_name="testflow3",
            hash_store_dir=self.hash_dir,
            logger_name="testlogger3",
        )

        orch.run(
            inputs=[self.input1],
            output_func=dummy_output_func,
            worker=dummy_worker,
            code_id="v1",
        )

        stored_hash_file_path = self.hash_dir / "testflow3.json"
        stored_hash_file = self.load_json_file(stored_hash_file_path)

        # Test input changed
        # Modify an input
        self.input1.write_text("AAB")

        orch.run(
            inputs=[self.input1],
            output_func=dummy_output_func,
            worker=dummy_worker,
            code_id="v1",
        )

        stored_hash_file_rerun = self.load_json_file(stored_hash_file_path)

        self.assertNotEqual(
            ordered(stored_hash_file),
            ordered(stored_hash_file_rerun),
            "The content is equal, despite an input change",
        )

        # Test output missing
        # Delete out
        #
        out1 = self.tempdir / "a_out1.txt"
        self.assertTrue(out1.exists())
        out1.unlink()
        self.assertFalse(out1.exists(), "Assertion failed: File was not successfully deleted.")
        orch.run(
            inputs=[self.input1],
            output_func=dummy_output_func,
            worker=dummy_worker,
            code_id="v1",
        )
        # test it is back
        self.assertTrue(out1.exists())

        # Test rerun with no change
        orch.run(
            inputs=[self.input1],
            output_func=dummy_output_func,
            worker=dummy_worker,
            code_id="v1",
        )

        stored_hash_file_rerun_nochange = self.load_json_file(stored_hash_file_path)

        self.assertEqual(
            ordered(stored_hash_file_rerun_nochange),
            ordered(stored_hash_file_rerun),
            "The content is not equal, even ignoring list order.",
        )

        # Test that the orchestrator loads the same hashes if redefined
        orch_2 = Orchestrator(
            workflow_name="testflow3",
            hash_store_dir=self.hash_dir,
            logger_name="testlogger3",
        )

        self.assertEqual(orch_2.hashes, orch.hashes)

    def test_rerun_when_code_id_changes(self):
        """Test that a change in code_id forces rerun."""

        orch = Orchestrator(
            workflow_name="testflow_codechange",
            hash_store_dir=self.hash_dir,
            logger_name="testlogger_codechange",
        )

        # First run with code_id = v1
        orch.run(
            inputs=[self.input1],
            output_func=dummy_output_func,
            worker=dummy_worker,
            code_id="v1",
        )

        # Load stored hash snapshot
        stored_hash_file_path = self.hash_dir / "testflow_codechange.json"
        hashes_before = self.load_json_file(stored_hash_file_path)

        # Now run with different code_id → MUST rerun due to changed code_hash
        orch.run(
            inputs=[self.input1],
            output_func=dummy_output_func,
            worker=dummy_worker,
            code_id="v2",
        )

        # Load updated hashes
        hashes_after = self.load_json_file(stored_hash_file_path)

        # They must differ (code_hash changed)
        self.assertNotEqual(
            ordered(hashes_before),
            ordered(hashes_after),
            "Hashes did not change despite code_id change.",
        )

    def test_worker_exception_is_logged_and_raised(self):
        """Test that an exception inside the worker hits the except block."""

        orch = Orchestrator(
            workflow_name="testflow_exception",
            hash_store_dir=self.hash_dir,
            logger_name="testlogger_exception",
        )

        hashfile = self.hash_dir / "testflow_exception.json"
        if hashfile.exists():
            hashfile.unlink()

        # The worker should raise, and orchestrator should re-raise
        with self.assertRaises(RuntimeError):
            orch.run(
                inputs=[self.input1],
                output_func=dummy_output_func,
                worker=failing_worker,  # <–– module-level function
                code_id="v1",
                parallelism=1,  # safer: only one worker process
            )

        # Should not save a hash file after failure
        self.assertFalse(
            hashfile.exists(),
            "Hash file should not exist because the workflow aborted due to the worker failure.",
        )


class TestOrchestratorWipe(unittest.TestCase):
    """Unit tests for the new wipe_dirs functionality in Orchestrator.run()."""

    def setUp(self):
        self.tempdir = Path(tempfile.mkdtemp())
        self.hash_dir = self.tempdir / "hashes"
        self.hash_dir.mkdir()

        # Create input files
        self.input1 = self.tempdir / "a.txt"
        self.input2 = self.tempdir / "b.txt"
        self.input1.write_text("AAA")
        self.input2.write_text("BBB")

        # Setup directories to wipe
        self.dir1 = self.tempdir / "dir1"
        self.dir2 = self.tempdir / "dir2"
        self.dir1.mkdir()
        self.dir2.mkdir()
        # Add some files in these dirs
        (self.dir1 / "old1.txt").write_text("keep me")
        (self.dir2 / "old2.txt").write_text("keep me too")

        self.orch = Orchestrator(
            workflow_name="wipe_helper_test",
            hash_store_dir=self.tempdir / "hashes",
        )

        self.code_file = self.orch._workflow_code_file()

    def test_wipe_contents_only_on_code_change(self):
        """Test that directory contents are wiped only when code_id changes and wipe_dirs is provided."""

        orch = Orchestrator(
            workflow_name="wipe_test_flow",
            hash_store_dir=self.hash_dir,
            logger_name="wipe_test_logger",
        )

        # Setup directories for outputs
        self.dir1.mkdir(exist_ok=True)
        self.dir2.mkdir(exist_ok=True)

        # Create initial outputs for input1 and input2
        (self.dir1 / "a_out1.txt").write_text("old1")
        (self.dir2 / "b_out2.txt").write_text("old2")

        # First run with code_id="v1" → initial wipe is expected
        orch.run(
            inputs=[self.input1, self.input2],
            output_func=dummy_output_func,
            worker=dummy_worker,
            code_id="v1",
            wipe_dirs=[self.dir1, self.dir2],
        )

        # After first run, old files are wiped, only worker outputs exist
        # Optional: you can check that manual files do not exist yet
        self.assertFalse(
            (self.dir1 / "a_out1.txt").read_text() == "old1"
            if (self.dir1 / "a_out1.txt").exists()
            else False
        )
        self.assertFalse(
            (self.dir2 / "b_out2.txt").read_text() == "old2"
            if (self.dir2 / "b_out2.txt").exists()
            else False
        )

        # Add manual files after first run
        (self.dir1 / "manual.txt").write_text("manual")
        (self.dir2 / "manual.txt").write_text("manual")

        # Run again with same code_id → nothing should be wiped
        orch.run(
            inputs=[self.input1, self.input2],
            output_func=dummy_output_func,
            worker=dummy_worker,
            code_id="v1",
            wipe_dirs=[self.dir1, self.dir2],
        )

        # All manual and worker-generated files should still exist
        self.assertTrue((self.dir1 / "manual.txt").exists())
        self.assertTrue((self.dir2 / "manual.txt").exists())

        # Run with new code_id → contents should be wiped
        orch.run(
            inputs=[self.input1, self.input2],
            output_func=dummy_output_func,
            worker=dummy_worker,
            code_id="v2",
            wipe_dirs=[self.dir1, self.dir2],
        )

        # Manual files and old worker outputs should be gone
        self.assertFalse((self.dir1 / "manual.txt").exists())
        self.assertFalse((self.dir2 / "manual.txt").exists())

        # The directories themselves must still exist
        self.assertTrue(self.dir1.exists())
        self.assertTrue(self.dir2.exists())

        # All per-file hashes should now have the new code_hash
        expected_code_hash = orch._hash_code_reference("v2")
        for record in orch.hashes.values():
            self.assertEqual(record["code_hash"], expected_code_hash)

    def test_no_wipe_if_code_id_unchanged(self):
        """Directory contents remain if code_id does not change."""
        orch = Orchestrator(
            workflow_name="wipe_test_nochange",
            hash_store_dir=self.hash_dir,
            logger_name="wipe_test_logger",
        )

        orch.run(
            inputs=[self.input1],
            output_func=dummy_output_func,
            worker=dummy_worker,
            code_id="v1",
            wipe_dirs=[self.dir1],
        )

        # Add a file
        (self.dir1 / "manual.txt").write_text("manual")
        self.assertTrue((self.dir1 / "manual.txt").exists())

        # Run again with same code_id → should NOT wipe
        orch.run(
            inputs=[self.input1],
            output_func=dummy_output_func,
            worker=dummy_worker,
            code_id="v1",
            wipe_dirs=[self.dir1],
        )

        # File should still exist
        self.assertTrue((self.dir1 / "manual.txt").exists())

    def test_no_wipe_when_wipe_dirs_none(self):
        """No wipe occurs if wipe_dirs=None even on code change."""
        orch = Orchestrator(
            workflow_name="wipe_test_none",
            hash_store_dir=self.hash_dir,
            logger_name="wipe_test_logger",
        )

        # Run first
        orch.run(
            inputs=[self.input1],
            output_func=dummy_output_func,
            worker=dummy_worker,
            code_id="v1",
            wipe_dirs=None,
        )

        # Add file to tempdir
        temp_file = self.dir1 / "manual.txt"
        temp_file.write_text("manual")
        self.assertTrue(temp_file.exists())

        # Run again with new code_id but wipe_dirs=None
        orch.run(
            inputs=[self.input1],
            output_func=dummy_output_func,
            worker=dummy_worker,
            code_id="v2",
            wipe_dirs=None,
        )

        # File should still exist
        self.assertTrue(temp_file.exists())

    def test_wipe_with_multiple_directories(self):
        """Wipes all directories provided in wipe_dirs list."""
        orch = Orchestrator(
            workflow_name="wipe_test_multi",
            hash_store_dir=self.hash_dir,
            logger_name="wipe_test_logger",
        )

        # Add files to dirs
        (self.dir1 / "file1.txt").write_text("file1")
        (self.dir2 / "file2.txt").write_text("file2")

        # First run
        orch.run(
            inputs=[self.input1],
            output_func=dummy_output_func,
            worker=dummy_worker,
            code_id="v1",
            wipe_dirs=[self.dir1, self.dir2],
        )

        # Add more files
        (self.dir1 / "file3.txt").write_text("file3")
        (self.dir2 / "file4.txt").write_text("file4")

        # Second run with new code_id → wipe should remove all
        orch.run(
            inputs=[self.input1],
            output_func=dummy_output_func,
            worker=dummy_worker,
            code_id="v2",
            wipe_dirs=[self.dir1, self.dir2],
        )

        # Check directories are empty
        self.assertEqual(list(self.dir1.iterdir()), [])
        self.assertEqual(list(self.dir2.iterdir()), [])

    def test_wipe_directory_contents_dir_does_not_exist(self):
        """Test that _wipe_directory_contents creates the directory if missing."""
        new_dir = self.tempdir / "new_dir"
        self.assertFalse(new_dir.exists(), "Directory should not exist yet")

        # Call helper
        self.orch._wipe_directory_contents(new_dir)

        # Check directory now exists and is empty
        self.assertTrue(new_dir.exists(), "Directory should have been created")
        self.assertEqual(list(new_dir.iterdir()), [], "Directory should be empty")

    def test_wipe_directory_contents_mixed_contents(self):
        """Test that _wipe_directory_contents removes both files and subdirectories."""
        test_dir = self.tempdir / "test_dir"
        test_dir.mkdir()

        # Create file
        file_path = test_dir / "file.txt"
        file_path.write_text("hello")

        # Create subdirectory with a file
        sub_dir = test_dir / "subdir"
        sub_dir.mkdir()
        sub_file = sub_dir / "subfile.txt"
        sub_file.write_text("world")

        # Sanity check before wiping
        self.assertTrue(file_path.exists())
        self.assertTrue(sub_dir.exists())
        self.assertTrue(sub_file.exists())

        # Call helper
        self.orch._wipe_directory_contents(test_dir)

        # Directory still exists but contents removed
        self.assertTrue(test_dir.exists(), "Directory itself should remain")
        self.assertEqual(list(test_dir.iterdir()), [], "Directory contents should be wiped")

    def test_wipe_directory_contents_empty_directory(self):
        """Test wiping a directory that exists but is already empty."""
        empty_dir = self.tempdir / "empty_dir"
        empty_dir.mkdir()

        # Sanity check: directory exists and is empty
        self.assertTrue(empty_dir.exists())
        self.assertEqual(list(empty_dir.iterdir()), [])

        # Call helper
        self.orch._wipe_directory_contents(empty_dir)

        # Directory should still exist and remain empty
        self.assertTrue(empty_dir.exists(), "Directory should still exist")
        self.assertEqual(list(empty_dir.iterdir()), [], "Directory should remain empty")

    def test_load_workflow_code_hash_with_invalid_json(self):
        """Trigger the except branch by writing malformed JSON."""
        self.code_file.parent.mkdir(parents=True, exist_ok=True)

        # Write invalid JSON
        self.code_file.write_text("{ invalid json }")

        # Call helper — should catch exception and return None
        result = self.orch._load_workflow_code_hash()
        self.assertIsNone(result, "Expected None when JSON is invalid")

    def test_load_workflow_code_hash_file_missing(self):
        """Trigger the return None branch when file does not exist."""
        # Ensure file does not exist
        if self.code_file.exists():
            self.code_file.unlink()

        result = self.orch._load_workflow_code_hash()
        self.assertIsNone(result, "Expected None when workflow code file is missing")


if __name__ == "__main__":
    unittest.main()
