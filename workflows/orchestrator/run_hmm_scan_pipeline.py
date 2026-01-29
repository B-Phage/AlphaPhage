#!/usr/bin/env python3
"""
Run HMM scan on a set of protein files using a specified YAML configuration.
Tracks per-file outputs in DuckDB and summary CSVs.
"""

import argparse
import hashlib
import subprocess
import sys
from functools import partial
from pathlib import Path

import pandas as pd
from orchestration.orchestrator import Orchestrator
from utils.config_loader import get_config  # YAML loader


# -----------------------------
# Top-level worker function
# -----------------------------
def run_hmm_worker(
    protein_file: Path,
    outputs: list,
    *,
    hmm: str,
    annotations: str,
    method: str,
    duckdb_path: Path,
):
    summary_csv = outputs[0]
    summary_csv.parent.mkdir(exist_ok=True, parents=True)

    cmd = [
        sys.executable,
        "workflows/cli/hmm_scan.py",
        "-f",
        protein_file,
        "-hp",
        hmm,
        "-a",
        annotations,
        "-m",
        method,
        "-o",
        summary_csv,
        "-db",
        duckdb_path,
    ]

    # subprocess.run(cmd, capture_output=True, text=True, check=True)
    subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL,
        # stderr is not defined, so it "inherits" the terminal
        check=True,
    )
    return outputs


# -----------------------------
# Top-level output function
# -----------------------------
def hmm_output_func(protein_file: Path, result_dir: Path):
    return [result_dir / f"{protein_file.stem}.csv"]


# -----------------------------
# Main entrypoint
# -----------------------------
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run HMM scan on protein files")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML config file for this run",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force rerun all protein files regardless of hash",
    )

    args = parser.parse_args(argv)

    # -----------------------------
    # Load configuration
    # -----------------------------
    gen_cfg = get_config(override_path=args.config, params_set="general")
    prot_cfg = get_config(override_path=args.config, params_set="prodigal_config")
    hmm_cfg = get_config(override_path=args.config, params_set="hmm_config")

    protein_dir = Path(prot_cfg.output_dir)
    result_dir = Path(hmm_cfg.output_dir)
    result_dir.mkdir(exist_ok=True, parents=True)
    duckdb_path = Path(hmm_cfg.duckdb_file)
    hash_store_dir = Path("ml_assets/data/workflow/hashes")

    # -----------------------------
    # Resolve input proteins
    # -----------------------------
    if gen_cfg.lst_genomes:
        protein_files = [protein_dir / f"{g}.fa" for g in gen_cfg.lst_genomes]
    else:
        protein_files = sorted(protein_dir.glob("*.fa"))

    # -----------------------------
    # Create pickleable worker/output using functools.partial
    # -----------------------------
    worker_fn = partial(
        run_hmm_worker,
        hmm=hmm_cfg.hmm_multifile,
        annotations=hmm_cfg.annotations,
        method=hmm_cfg.hmm_scan_method,
        duckdb_path=duckdb_path,
    )

    output_fn = partial(hmm_output_func, result_dir=result_dir)

    # -----------------------------
    # Build orchestrator
    # -----------------------------
    orch = Orchestrator(
        workflow_name=f"hmm_scan_{hmm_cfg.hmm_scan_method}",
        hash_store_dir=hash_store_dir,
        logger_name="hmm_scan",
    )

    # Add config hash to code ID to trigger reruns on config change
    with open(args.config, "rb") as f:
        config_hash = hashlib.sha256(f.read()).hexdigest()

    full_code_id = f"{gen_cfg.code_version}_{config_hash}"

    orch.run(
        inputs=protein_files,
        output_func=output_fn,
        worker=worker_fn,
        code_id=full_code_id,
        parallelism=gen_cfg.n_jobs,
        force=args.force,
        tqdm_desc=f"HMM scanning ({hmm_cfg.hmm_scan_method})",
    )


# -----------------------------
# CLI hook
# -----------------------------
if __name__ == "__main__":
    main()
