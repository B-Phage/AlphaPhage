#!/usr/bin/env python3

import argparse
import pathlib
from functools import partial
from pathlib import Path

from hmm.duckdb_io import (
    extract_hmm_sequences_for_msa,
    get_distinct_hmms,
)
from orchestration.orchestrator import Orchestrator
from utils.config_loader import get_config
from utils.file_system import clear_dir
from utils.hash import create_manifest

# --- Path Anchoring for default internal assets ---
SCRIPT_PATH = pathlib.Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parent.parent
DEFAULT_RUNTIME_CFG = REPO_ROOT / "ml_assets" / "config" / "runtime.yaml"
DEFAULT_HASH_DIR = REPO_ROOT / "ml_assets" / "data" / "workflow" / "hashes"
DEFAULT_HMM_ID_DIR = REPO_ROOT / "ml_assets" / "data" / "workflow" / "hmm_ids"

# ---------------------------------------------------------
# Top-Level Worker Functions (Pickleable)
# ---------------------------------------------------------


def hmm_fasta_worker(input_path: Path, outputs: list, db_path: Path, table_name: str):
    """Worker logic at the top level so multiprocessing can find it."""
    hmm_name = input_path.read_text().strip()
    out_fasta = outputs[0]

    extract_hmm_sequences_for_msa(
        db_path=db_path,
        table_name=table_name,
        hmm_domain_name=hmm_name,
        out_fasta=out_fasta,
    )


def hmm_fasta_output_func(input_path: Path, fasta_root: Path):
    """Output mapping logic at the top level."""
    hmm_name = input_path.read_text().strip()
    return [fasta_root / f"{hmm_name}.fasta"]


# ---------------------------------------------------------
# Workflow Orchestration
# ---------------------------------------------------------


def run_workflow(config_path, hash_store_dir=None, hmm_id_dir=None, force=False):
    # 1. Load Configs
    gen_cfg = get_config(override_path=config_path, params_set="general")
    cfg = get_config(override_path=config_path, params_set="hmm_config")
    run_cfg = get_config(str(DEFAULT_RUNTIME_CFG), params_set="meta")

    # 2. Resolve Paths
    DB_PATH = Path(cfg.duckdb)
    TABLE_NAME = cfg.summary_tbl
    FASTA_ROOT = Path(cfg.hmm_fasta_dir)

    # Logic: CLI Argument > Repo Default
    hash_dir = Path(hash_store_dir) if hash_store_dir else DEFAULT_HASH_DIR
    id_dir = Path(hmm_id_dir) if hmm_id_dir else DEFAULT_HMM_ID_DIR

    # 3. Setup Directories
    FASTA_ROOT.mkdir(parents=True, exist_ok=True)
    id_dir.mkdir(parents=True, exist_ok=True)

    clear_dir(FASTA_ROOT)
    clear_dir(id_dir)

    # 4. Prepare Inputs
    hmms = get_distinct_hmms(DB_PATH, TABLE_NAME)
    hmm_id_files = []
    for hmm in hmms:
        p = id_dir / f"{hmm}.hmm_id"
        p.write_text(hmm)
        hmm_id_files.append(p)

    # 5. Create Pickleable Worker/Output using partial
    # This 'bakes in' the paths while keeping the function at the top level
    worker_fn = partial(hmm_fasta_worker, db_path=DB_PATH, table_name=TABLE_NAME)

    output_fn = partial(hmm_fasta_output_func, fasta_root=FASTA_ROOT)

    # 6. Run Orchestrator
    orch = Orchestrator(
        workflow_name=f"export_{gen_cfg.get('name', 'test')}_hmm_fastas",
        hash_store_dir=hash_dir,
        logger_name="hmm_fasta_export",
    )

    orch.run(
        inputs=hmm_id_files,
        output_func=output_fn,
        worker=worker_fn,
        code_id=gen_cfg.code_version,
        parallelism=run_cfg.n_jobs,
        force=force,
        tqdm_desc=f"Exporting {gen_cfg.get('name', 'test')} HMM FASTAs",
    )

    # 7. Manifest Generation
    fasta_files = list(FASTA_ROOT.glob("*.fasta"))
    # Handle pathlib object string concatenation for manifest file
    manifest_path = Path(str(cfg.hmm_fasta_dir) + "_manifest.json")
    create_manifest(fasta_files, manifest_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export per-HMM FASTA files.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--hash_store", help="Optional override for metadata hashes")
    parser.add_argument("--hmm_id_dir", help="Optional override for temporary HMM ID files")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    run_workflow(
        config_path=args.config,
        hash_store_dir=args.hash_store,
        hmm_id_dir=args.hmm_id_dir,
        force=args.force,
    )
