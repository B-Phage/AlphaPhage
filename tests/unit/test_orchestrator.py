import json
import tempfile
import unittest
from pathlib import Path

from orchestration.orchestrator import Orchestrator


def dummy_worker(input_path: Path, outputs: list):
    for out in outputs:
        out.write_text(f"Processed {input_path.read_text()}")


def dummy_output_func(input_path: Path):
    return [
        input_path.parent / f"{input_path.stem}_out1.txt",
        input_path.parent / f"{input_path.stem}_out2.txt",
    ]


class TestOrchestrator(unittest.TestCase):

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


if __name__ == "__main__":
    unittest.main()
