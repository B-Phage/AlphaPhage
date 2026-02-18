#!/usr/bin/env python3
"""
Run HMM scan on a set of protein files using a specified YAML configuration.
Tracks per-file outputs in a parquet data lake and summary CSVs.
"""

import argparse
import hashlib
from functools import partial
from pathlib import Path

from hmm.api import run_hmm_scan_on_file
from orchestration.orchestrator import Orchestrator
from utils.config_loader import get_config  # YAML loader
from utils.hash import compute_combined_code_version, create_manifest


# -----------------------------
# Top-level worker function
# -----------------------------
def run_hmm_worker(
    protein_file: Path,
    outputs: list,
    *,
    hmm: Path,
    annotations: Path,
    method: str,
    data_lake: Path,
):
    # Make sure the data lake folder exists
    data_lake.mkdir(parents=True, exist_ok=True)

    summary_csv = outputs[0]
    summary_csv.parent.mkdir(exist_ok=True, parents=True)
    csv_file = summary_csv / f"{protein_file.stem}.csv"

    run_hmm_scan_on_file(
        in_fasta=protein_file,
        hmm_path=hmm,
        annotation_file=annotations,
        method=method,
        out_csv=csv_file,
        out_database=data_lake,
    )

    # Return the folder as the "output" for the orchestrator
    return [data_lake]


# -----------------------------
# Top-level output function
# -----------------------------
def hmm_output_func(protein_file: Path, result_dir: Path):
    # All protein files contribute to the same data lake
    return [result_dir]


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
        "--code_hash",
        action="append",
        type=Path,
        required=True,
        help="Path to a hash file (repeatable)",
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
    run_cfg = get_config(
        "ml_assets/config/runtime.yaml", params_set="meta"
    )  # config that does not affect outputs
    prot_cfg = get_config(override_path=args.config, params_set="prodigal_config")
    hmm_cfg = get_config(override_path=args.config, params_set="hmm_config")

    protein_dir = Path(prot_cfg.proteins_dir)
    result_dir = Path(hmm_cfg.hmm_scan_dir)
    result_dir.mkdir(exist_ok=True, parents=True)
    data_lake = Path(hmm_cfg.data_lake)
    data_lake.mkdir(parents=True, exist_ok=True)
    hash_store_dir = Path("ml_assets/data/workflow/hashes")

    # Add code hash to trigger reruns on code or config changes
    CONF_CODE_VERSION = gen_cfg.code_version
    CODE_VERSION = compute_combined_code_version(
        base_version=CONF_CODE_VERSION,
        hash_files=args.code_hash,
        config_items=[gen_cfg.name, hmm_cfg, protein_dir, data_lake],
    )

    # -----------------------------
    # Resolve input proteins
    # -----------------------------
    print(f"Protein directory: {protein_dir}")
    if gen_cfg.lst_genomes:
        protein_files = [protein_dir / f"{g}.fa" for g in gen_cfg.lst_genomes]
    else:
        protein_files = sorted(protein_dir.glob("*.fa"))

    print(f"Protein files: {protein_files}")

    # -----------------------------
    # Create pickleable worker/output using functools.partial
    # -----------------------------
    worker_fn = partial(
        run_hmm_worker,
        hmm=hmm_cfg.hmm_multifile,
        annotations=hmm_cfg.annotations,
        method=hmm_cfg.hmm_scan_method,
        data_lake=data_lake,
    )

    output_fn = partial(hmm_output_func, result_dir=result_dir)

    # -----------------------------
    # Build orchestrator
    # -----------------------------
    orch = Orchestrator(
        workflow_name=f"hmm_scan_{gen_cfg.get('name', 'test')}",
        hash_store_dir=hash_store_dir,
        logger_name="hmm_scan",
    )

    orch.run(
        inputs=protein_files,
        output_func=output_fn,
        worker=worker_fn,
        code_id=CODE_VERSION,
        parallelism=run_cfg.n_jobs,
        force=args.force,
        wipe_dirs=[result_dir, data_lake],
        tqdm_desc=f"HMM scanning ({gen_cfg.get('name', 'test')})",
    )

    # create the manifest file for DVC pipeline tracking
    parquet_files = list(data_lake.rglob("*.parquet"))
    MANIFEST_FILE = Path(hmm_cfg.data_lake + "_manifest.json")
    create_manifest(parquet_files, MANIFEST_FILE)


# -----------------------------
# CLI hook
# -----------------------------
if __name__ == "__main__":
    main()
