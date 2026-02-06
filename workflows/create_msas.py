#!/usr/bin/env python3
"""
Workflow: Multiple Sequence Alignment (MSA) Generation
Processes HMM hits into CSA (Compressed Sequence Alignment) format using
orchestrated parallel processing.
"""

import argparse
import os
from functools import partial
from pathlib import Path

from hmm.msa import make_msa
from orchestration.orchestrator import Orchestrator
from utils.config_loader import get_config

# -----------------------------------------------------------------------------
# Multiprocessing Workers (Must be top-level for 'spawn' pickling)
# -----------------------------------------------------------------------------


def msa_worker(input_path: Path, outputs: list, split_path: Path):
    """
    Executes the MSA generation for a single HMM fasta file.

    Args:
        input_path: Path to the input .fasta file.
        outputs: List containing the target output path (from output_func).
        split_path: Directory containing the reference .hmm files.
    """
    hmm_name = input_path.stem
    out_csa = outputs[0]
    individual_hmm = split_path / f"{hmm_name}.hmm"

    make_msa(hmm_seq_hits_file=input_path, hmm_file_path=individual_hmm, out_msa_path=out_csa)


def get_msa_outputs(input_path: Path, output_dir: Path) -> list[Path]:
    """
    Maps an input fasta file to its expected .csa output path.
    Used by the Orchestrator to check for existing results.
    """
    hmm_name = input_path.stem
    return [output_dir / f"{hmm_name}.csa"]


# -----------------------------------------------------------------------------
# Workflow Orchestration
# -----------------------------------------------------------------------------


def run_msa_workflow(config_path: str, force: bool = False):
    """
    Orchestrates the MSA generation process.

    Loads configuration, prepares the file manifest, and executes the
    parallel processing loop via the Orchestrator.
    """
    # Load configuration sets
    run_cfg = get_config("ml_assets/config/runtime.yaml", params_set="meta")
    gen_cfg = get_config(override_path=config_path, params_set="general")
    hmm_cfg = get_config(override_path=config_path, params_set="hmm_config")
    msa_cfg = get_config(override_path=config_path, params_set="msa_config")

    # Path resolution
    HMM_FILE_DIR = Path(hmm_cfg.hmm_fasta_dir)
    SPLIT_PATH = Path(msa_cfg.split_hmm_dir)
    OUTPUT_DIR = Path(msa_cfg.msa_csa_dir)
    HASH_STORE_DIR = Path("ml_assets/data/workflow/hashes")

    # Setup environment
    hmm_fasta_files = sorted(HMM_FILE_DIR.glob("*.fasta"))
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Partial functions allow passing extra context (paths) to workers
    # while remaining picklable for multiprocessing.
    worker_with_args = partial(msa_worker, split_path=SPLIT_PATH)
    output_func_with_args = partial(get_msa_outputs, output_dir=OUTPUT_DIR)

    orch = Orchestrator(
        workflow_name=f"msa_{hmm_cfg.hmm_scan_method}",
        hash_store_dir=HASH_STORE_DIR,
        logger_name="msa",
    )

    orch.run(
        inputs=hmm_fasta_files,
        output_func=output_func_with_args,
        worker=worker_with_args,
        code_id=gen_cfg.code_version,
        parallelism=run_cfg.n_jobs,
        force=force,
        tqdm_desc=f"MSA processing ({hmm_cfg.hmm_scan_method})",
    )


# -----------------------------------------------------------------------------
# CLI Entry Point
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate MSAs from protein-HMM hit files.")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the YAML configuration file."
    )
    parser.add_argument(
        "--force", action="store_true", help="Bypass hash checks and rerun all inputs."
    )

    args = parser.parse_args()

    run_msa_workflow(config_path=args.config, force=args.force)
