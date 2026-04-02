#!/usr/bin/env python3
"""
Run HMM scan on a set of protein files using a specified YAML configuration.
Tracks per-file outputs in a parquet data lake and summary CSVs.
"""

import argparse
import pathlib
from functools import partial
from pathlib import Path

import utils.consol_log as log
from hmm.api import run_hmm_scan_on_file
from orchestration.orchestrator import Orchestrator
from utils.config_loader import get_config
from utils.hash import compute_combined_code_version, create_manifest

logger = log.get_logger()

# --- Path Anchoring ---
SCRIPT_PATH = pathlib.Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parent.parent
DEFAULT_RUNTIME_CFG = REPO_ROOT / "ml_assets" / "config" / "runtime.yaml"
DEFAULT_HASH_DIR = REPO_ROOT / "ml_assets" / "data" / "workflow" / "hashes"

# -------------------------------------------------------------
# Top-level worker functions (Pickleable for Multiprocessing)
# -------------------------------------------------------------


def run_hmm_worker(
    protein_file: Path,
    outputs: list,
    *,
    hmm: Path,
    annotations: Path,
    method: str,
    data_lake: Path,
):
    """Worker logic at the top level for pickling."""
    # Ensure local data lake folder exists for this specific worker
    data_lake.mkdir(parents=True, exist_ok=True)

    csv_file = Path(outputs[0])
    csv_file.parent.mkdir(exist_ok=True, parents=True)

    run_hmm_scan_on_file(
        in_fasta=protein_file,
        hmm_path=hmm,
        annotation_file=annotations,
        method=method,
        out_csv=csv_file,
        out_database=data_lake,
    )

    return [data_lake]


def hmm_output_func(protein_file: Path, result_dir: Path):
    """Output mapping logic at the top level."""
    return [result_dir / f"{protein_file.stem}.csv"]


# -------------------------------------------------------------
# Workflow Orchestration
# -------------------------------------------------------------


def run_workflow(config_path, code_hash_files, hash_store_dir=None, force=False):
    # 1. Load configuration
    gen_cfg = get_config(override_path=config_path, params_set="general")
    run_cfg = get_config(str(DEFAULT_RUNTIME_CFG), params_set="meta")
    prot_cfg = get_config(override_path=config_path, params_set="prodigal_config")
    hmm_cfg = get_config(override_path=config_path, params_set="hmm_config")

    # 2. Resolve Paths
    protein_dir = Path(prot_cfg.proteins_dir)
    result_dir = Path(hmm_cfg.hmm_scan_dir)
    data_lake = Path(hmm_cfg.data_lake)

    # Logic: CLI Argument > Repo Default
    hash_dir = Path(hash_store_dir) if hash_store_dir else DEFAULT_HASH_DIR

    # 3. Setup Directories
    result_dir.mkdir(exist_ok=True, parents=True)
    data_lake.mkdir(parents=True, exist_ok=True)

    # 4. Code Versioning
    # Combined hash of logic version, config items, and paths
    CODE_VERSION = compute_combined_code_version(
        base_version=gen_cfg.code_version,
        hash_files=code_hash_files,
        config_items=[gen_cfg.name, hmm_cfg, protein_dir, data_lake],
    )

    # 5. Resolve input proteins
    logger.info(f"Protein directory: {protein_dir}")
    if gen_cfg.lst_genomes:
        protein_files = [protein_dir / f"{g}.fa" for g in gen_cfg.lst_genomes]
    else:
        protein_files = sorted(protein_dir.glob("*.fa"))
    logger.info(f"Protein files found: {len(protein_files)}")

    # 6. Create pickleable worker/output using functools.partial
    worker_fn = partial(
        run_hmm_worker,
        hmm=Path(hmm_cfg.hmm_multifile),
        annotations=Path(hmm_cfg.annotations),
        method=hmm_cfg.hmm_scan_method,
        data_lake=data_lake,
    )

    output_fn = partial(hmm_output_func, result_dir=result_dir)

    # 7. Build and execute orchestrator
    orch = Orchestrator(
        workflow_name=f"hmm_scan_{gen_cfg.get('name', 'test')}",
        hash_store_dir=hash_dir,
        logger_name="hmm_scan",
    )

    orch.run(
        inputs=protein_files,
        output_func=output_fn,
        worker=worker_fn,
        code_id=CODE_VERSION,
        parallelism=run_cfg.n_jobs,
        force=force,
        wipe_dirs=[result_dir, data_lake],
        tqdm_desc=f"HMM scanning ({gen_cfg.get('name', 'test')})",
    )

    # 8. Manifest Generation for DVC/Inference tracking
    parquet_files = list(data_lake.rglob("*.parquet"))
    manifest_path = Path(str(hmm_cfg.data_lake) + "_manifest.json")
    create_manifest(parquet_files, manifest_path)
    logger.info(f"✅ HMM Scan Complete. Manifest: {manifest_path}")


# -----------------------------
# Main CLI Entrypoint
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run HMM scan on protein files")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--code_hash", action="append", type=Path, required=True)
    parser.add_argument("--hash_store", help="Optional override for metadata hashes")
    parser.add_argument("--force", action="store_true")

    args = parser.parse_args()

    run_workflow(
        config_path=args.config,
        code_hash_files=args.code_hash,
        hash_store_dir=args.hash_store,
        force=args.force,
    )
