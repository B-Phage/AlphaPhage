import hashlib
import json
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import utils.consol_log as log
from tqdm import tqdm
from utils.hash import hash_file, hash_string


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
        hash_store_dir: Path = Path("ml_assets/data/workflow/hashes"),
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
        return hash_file(file)

    def _hash_code_reference(self, code_id: str) -> str:
        """
        Allows tying input results to a code version.
        Code version may be a git SHA, a module checksum, or a string you supply.
        """
        return hash_string(code_id)

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

    def _wipe_directory_contents(self, directory: Path) -> None:
        """
        Remove all contents of a directory but keep the directory itself.
        """

        if not directory.exists():
            directory.mkdir(parents=True, exist_ok=True)
            return

        for item in directory.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()

    # -------------------------------------------------------------

    def _workflow_code_file(self) -> Path:
        return self.hash_store.parent / f"{self.workflow_name}_code.json"

    # def _load_workflow_code_hash(self) -> str | None: #py3.10+ only
    def _load_workflow_code_hash(self) -> Optional[str]:
        file = self._workflow_code_file()
        if file.exists():
            try:
                with open(file, "r") as f:
                    return json.load(f).get("code_hash")
            except Exception:
                return None
        return None

    def _save_workflow_code_hash(self, code_hash: str) -> None:
        with open(self._workflow_code_file(), "w") as f:
            json.dump({"code_hash": code_hash}, f, indent=2)

    # -------------------------------------------------------------

    # wipe_dirs: list[Path] | None = None,
    def run(
        self,
        inputs: list[Path],
        output_func: Callable[[Path], list[Path]],
        worker: Callable[[Path, list[Path]], Any],
        code_id: str,
        parallelism: int = 4,
        force: bool = False,
        tqdm_desc: str = "Processing",
        wipe_dirs: Optional[list[Path]] = None,
    ):
        """
        Execute the workflow on a set of inputs with optional per-file skipping
        and conditional directory content wipe on code changes.

        This method orchestrates the processing of inputs using a worker function
        and tracks per-file input and code hashes to skip unchanged items. If a
        code change is detected (via `code_id`), it can wipe the contents of
        specified directories before processing to remove orphaned outputs.

        Args:
            inputs (list[Path]): List of input files to process.
            output_func (Callable[[Path], list[Path]]): Function that takes an input
                file and returns a list of output paths associated with it.
            worker (Callable[[Path, list[Path]], Any]): Function that performs
                processing on a single input file and writes outputs to the
                corresponding paths.
            code_id (str): String identifying the current code/config version. Changes
                in this ID trigger a wipe of `wipe_dirs` contents if provided.
            parallelism (int, optional): Maximum number of parallel worker processes.
                Defaults to 4.
            force (bool, optional): If True, force rerun all inputs regardless of
                hash comparison. Defaults to False.
            tqdm_desc (str, optional): Description for the tqdm progress bar. Defaults
                to "Processing".
            wipe_dirs (list[Path] | None, optional): Directories whose **contents**
                should be wiped if a code change is detected. Defaults to None
                (no wipe).

        Raises:
            Exception: Any exception raised by the worker function during processing
                will propagate and stop the workflow.

        Notes:
            - Only wipes **contents** of directories, not the directories themselves.
            - Per-file hashes are reset when a code change triggers a wipe.
            - DVC-managed outputs with `persist: true` are preserved as long as
            `wipe_dirs` does not include their parent directory.
            - Workflow-level code hash is saved after processing to detect future
            code changes.
        """

        code_hash = self._hash_code_reference(code_id)

        # -------------------------------------------------
        # Conditional wipe: only if code has changed
        # -------------------------------------------------
        previous_code_hash = self._load_workflow_code_hash()
        code_changed = previous_code_hash != code_hash

        if wipe_dirs and code_changed:
            for d in wipe_dirs:
                self._wipe_directory_contents(d)

        # ------------------------------------------------------------------
        # 2️⃣  Determine which inputs need processing
        # ------------------------------------------------------------------
        inputs_to_run = []

        for input_file in inputs:
            input_hash = self._hash_file(input_file)
            output_paths = output_func(input_file)

            if self._needs_rerun(
                input_file=input_file,
                output_paths=output_paths,
                input_hash=input_hash,
                code_hash=code_hash,
                force=force,
            ):
                inputs_to_run.append((input_file, input_hash, output_paths))

        self.logger.info(f"{len(inputs_to_run)} / {len(inputs)} inputs will be processed.")

        # ------------------------------------------------------------------
        # 3️⃣  Parallel execution
        # ------------------------------------------------------------------
        if inputs_to_run:
            with ProcessPoolExecutor(max_workers=parallelism) as executor:
                futures = {
                    executor.submit(worker, inp, outs): (inp, input_hash)
                    for inp, input_hash, outs in inputs_to_run
                }

                for future in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc=tqdm_desc,
                ):
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

        # ------------------------------------------------------------------
        # 4️⃣  Persist updated hashes + workflow code hash
        # ------------------------------------------------------------------
        self._save_hashes()
        self._save_workflow_code_hash(code_hash)

        self.logger.info("Workflow completed successfully.")
