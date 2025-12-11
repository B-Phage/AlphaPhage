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


if __name__ == "__main__":
    unittest.main()
