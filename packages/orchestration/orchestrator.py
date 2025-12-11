import hashlib
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import utils.consol_log as log
from tqdm import tqdm


class Orchestrator:
    """
    Generic Orchestrator to run per-file tasks with:
        - hashing for change detection
        - automatic skipping of unchanged items
        - parallel execution
        - support for multiple outputs
        - pluggable worker functions
        - uses external logger setup
    """

    def __init__(
        self,
        workflow_name: str,
        hash_store_dir: str = "ml_assets/data/workflow/hashes",
        logger_name: Optional[str] = None,
        log_level=log.INFO,
    ):
        self.workflow_name = workflow_name
        self.hash_store = Path(hash_store_dir) / f"{workflow_name}.json"
        self.hash_store.parent.mkdir(parents=True, exist_ok=True)

        self.hashes: Dict[str, Dict[str, str]] = self._load_hashes()

        # Setup logger
        self.logger = log.get_logger(
            level=log_level,
            name=logger_name if logger_name else f"orchestrator.{workflow_name}",
        )

    # -------------------------------------------------------------

    def _load_hashes(self) -> Dict:
        if self.hash_store.exists():
            with open(self.hash_store, "r") as f:
                return json.load(f)
        return {}

    def _save_hashes(self) -> None:
        with open(self.hash_store, "w") as f:
            json.dump(self.hashes, f, indent=2)

    # -------------------------------------------------------------

    def _hash_file(self, file: Path) -> str:
        """Return SHA256 hash of a file."""
        h = hashlib.sha256()
        with open(file, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    def _hash_code_reference(self, code_id: str) -> str:
        """
        Allows tying input results to a code version.
        Code version may be a git SHA, a module checksum, or a string you supply.
        """
        return hashlib.sha256(code_id.encode()).hexdigest()

    # -------------------------------------------------------------

    def _needs_rerun(
        self,
        input_file: Path,
        output_paths: List[Path],
        input_hash: str,
        code_hash: str,
        force: bool,
    ) -> bool:

        key = str(input_file)

        if force:
            self.logger.debug(f"Forcing rerun for {input_file}")
            return True

        if key not in self.hashes:
            self.logger.debug(f"No previous hash found for {input_file}")
            return True

        stored = self.hashes[key]

        # Check input hash
        if stored["input_hash"] != input_hash:
            self.logger.debug(f"Input changed for {input_file}")
            return True

        # Check code hash
        if stored["code_hash"] != code_hash:
            self.logger.debug(f"Code reference changed for {input_file}")
            return True

        # Check if outputs exist
        for out in output_paths:
            if not out.exists():
                self.logger.debug(f"Missing output {out} for {input_file}")
                return True

        return False

    # -------------------------------------------------------------

    def _update_hash_record(self, input_file: Path, input_hash: str, code_hash: str):
        self.hashes[str(input_file)] = {
            "input_hash": input_hash,
            "code_hash": code_hash,
        }

    # -------------------------------------------------------------

    def run(
        self,
        inputs: List[Path],
        output_func: Callable[[Path], List[Path]],
        worker: Callable[[Path, List[Path]], Any],
        code_id: str,
        parallelism: int = 4,
        force: bool = False,
        tqdm_desc: str = "Processing",
    ):
        """
        Run workflow:
            inputs: list of input paths
            output_func(input) -> list of output paths
            worker(input, outputs) -> produces output files
            code_id: string identifying code version
        """

        code_hash = self._hash_code_reference(code_id)

        inputs_to_run = []
        for input_file in inputs:
            input_hash = self._hash_file(input_file)
            output_paths = output_func(input_file)

            if self._needs_rerun(input_file, output_paths, input_hash, code_hash, force):
                inputs_to_run.append((input_file, input_hash, output_paths))

        self.logger.info(f"{len(inputs_to_run)} / {len(inputs)} inputs will be processed.")

        # Parallel execution
        with ProcessPoolExecutor(max_workers=parallelism) as executor:
            futures = {
                executor.submit(worker, inp, outs): (inp, input_hash)
                for inp, input_hash, outs in inputs_to_run
            }

            for future in tqdm(as_completed(futures), total=len(futures), desc=tqdm_desc):
                input_file, input_hash = futures[future]

                try:
                    future.result()
                    self._update_hash_record(
                        input_file=input_file,
                        input_hash=input_hash,
                        code_hash=code_hash,
                    )
                except Exception as e:
                    self.logger.error(f"Error processing {input_file}: {e}")
                    raise

        # Save updated hashes
        self._save_hashes()

        self.logger.info("Workflow completed successfully.")
