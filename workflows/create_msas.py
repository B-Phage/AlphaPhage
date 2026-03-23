#!/usr/bin/env python3
"""
Workflow: Multiple Sequence Alignment (MSA) Generation
Processes HMM hits into CSA (Compressed Sequence Alignment) format using
orchestrated parallel processing.
"""

import argparse
import os
import pathlib
from functools import partial
from pathlib import Path

from hmm.msa import make_msa
from orchestration.orchestrator import Orchestrator
from utils.config_loader import get_config
from utils.hash import compute_combined_code_version, create_manifest

# --- Path Anchoring ---
SCRIPT_PATH = pathlib.Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parent.parent
DEFAULT_RUNTIME_PATH = REPO_ROOT / "ml_assets" / "config" / "runtime.yaml"
DEFAULT_HASH_DIR = REPO_ROOT / "ml_assets" / "data" / "workflow" / "hashes"

# -----------------------------------------------------------------------------
# Multiprocessing Workers (Must be top-level for 'spawn' pickling)
# -----------------------------------------------------------------------------


def msa_worker(input_path: Path, outputs: list, split_path: Path):
    """
    Executes the MSA generation for a single HMM fasta file.
    """
    hmm_name = input_path.stem
    out_csa = outputs[0]
    individual_hmm = split_path / f"{hmm_name}.hmm"

    # Ensure output directory for this worker exists
    out_csa.parent.mkdir(parents=True, exist_ok=True)

    make_msa(hmm_seq_hits_file=input_path, hmm_file_path=individual_hmm, out_msa_path=out_csa)


def get_msa_outputs(input_path: Path, output_dir: Path) -> list[Path]:
    """
    Maps an input fasta file to its expected .csa output path.
    """
    hmm_name = input_path.stem
    return [output_dir / f"{hmm_name}.csa"]


# -----------------------------------------------------------------------------
# Workflow Orchestration
# -----------------------------------------------------------------------------


def run_msa_workflow(config_path: str, code_hash, hash_store_dir=None, force: bool = False):
    """
    Orchestrates the MSA generation process.
    """
    # 1. Load configuration
    run_cfg = get_config(str(DEFAULT_RUNTIME_PATH), params_set="meta")
    gen_cfg = get_config(override_path=config_path, params_set="general")
    hmm_cfg = get_config(override_path=config_path, params_set="hmm_config")
    msa_cfg = get_config(override_path=config_path, params_set="msa_config")

    # 2. Resolve Paths
    hmm_file_dir = Path(hmm_cfg.hmm_fasta_dir)
    split_path = Path(msa_cfg.split_hmm_dir)
    output_dir = Path(msa_cfg.msa_csa_dir)

    # Logic: CLI Override > Repo Default
    hash_dir = Path(hash_store_dir) if hash_store_dir else DEFAULT_HASH_DIR

    # 3. Code Versioning
    code_version = compute_combined_code_version(
        base_version=gen_cfg.code_version,
        hash_files=code_hash,
        config_items=[hmm_cfg.hmm_fasta_dir, msa_cfg],
    )

    # 4. Setup environment
    hmm_fasta_files = sorted(hmm_file_dir.glob("*.fasta"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # 5. Partial functions for picklable workers
    worker_with_args = partial(msa_worker, split_path=split_path)
    output_func_with_args = partial(get_msa_outputs, output_dir=output_dir)

    # 6. Run orchestrator
    orch = Orchestrator(
        workflow_name=f"msa_{gen_cfg.get('name', 'test')}",
        hash_store_dir=hash_dir,
        logger_name="msa",
    )

    orch.run(
        inputs=hmm_fasta_files,
        output_func=output_func_with_args,
        worker=worker_with_args,
        code_id=code_version,
        parallelism=run_cfg.n_jobs,
        force=force,
        tqdm_desc=f"MSA processing ({gen_cfg.get('name', 'test')})",
        wipe_dirs=[output_dir],
    )

    # 7. Manifest Generation (Looking for .csa files)
    msa_results = list(output_dir.glob("*.csa"))
    manifest_path = Path(str(msa_cfg.msa_csa_dir) + "_manifest.json")
    create_manifest(msa_results, manifest_path)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate MSAs from protein-HMM hit files.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--code_hash", action="append", type=Path, required=True)
    parser.add_argument("--hash_store", help="Optional override for metadata hashes")
    parser.add_argument("--force", action="store_true")

    args = parser.parse_args(argv)

    run_msa_workflow(
        config_path=args.config,
        code_hash=args.code_hash,
        hash_store_dir=args.hash_store,
        force=args.force,
    )


if __name__ == "__main__":
    main()
