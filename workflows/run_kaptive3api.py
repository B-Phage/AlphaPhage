#!/usr/bin/env python3
import argparse
import os
from functools import partial
from pathlib import Path

import utils.consol_log as log
from kaptive3api.kaptive3api_functions import kapsid_output_func, run_kapsid
from kaptive.database import load_database
from orchestration.orchestrator import Orchestrator
from utils.config_loader import get_config
from utils.hash import compute_combined_code_version, create_manifest

logger = log.get_logger()


def run_kaptive_workflow(
    config_path: str,
    code_hash_files: list[Path],
    hash_store: Path | None = None,
    force: bool = False,
):
    """
    Main entry point for the Kaptive serotyping workflow.
    """
    # 1. Configuration & Path Anchoring
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent

    run_cfg = get_config(str(repo_root / "ml_assets/config/runtime.yaml"), params_set="meta")
    gen_cfg = get_config(override_path=config_path, params_set="general")
    cfg = get_config(override_path=config_path, params_set="kaptive_config")

    # Resolve HASH_STORE
    hash_store_dir = hash_store if hash_store else (repo_root / "ml_assets/data/workflow/hashes")

    # 2. Strict Database Resolution
    # We expect the snapshot logic to have provided a valid path in the config
    orgdb_path = Path(cfg.orgdb_full_path).resolve()

    if not orgdb_path.exists():
        raise FileNotFoundError(
            f"❌ Kaptive Database not found at: {orgdb_path}\n"
            f"Check 'orgdb_full_path' in {config_path}"
        )

    logger.info(f"📚 Using Kaptive Database: {orgdb_path}")

    # 3. Versioning
    code_version = compute_combined_code_version(
        base_version=gen_cfg.code_version,
        hash_files=code_hash_files,
        config_items=[cfg],
    )

    # 4. Prepare Orchestrator Inputs
    genome_dir = Path(cfg.genome_dir).resolve()
    output_dir = Path(cfg.kaptive_dir).resolve()

    if not genome_dir.exists():
        raise FileNotFoundError(f"❌ Genome directory not found: {genome_dir}")

    genome_files = sorted(genome_dir.glob("*.fasta"))
    if not genome_files:
        logger.warning(f"⚠️ No .fasta files found in {genome_dir}")

    # Load database once for the worker
    db = load_database(str(orgdb_path))

    output_dir.mkdir(parents=True, exist_ok=True)
    bound_worker = partial(run_kapsid, db_obj=db)
    bound_output = partial(kapsid_output_func, output_dir=output_dir)

    # 5. Execution
    orch = Orchestrator(
        workflow_name=f"kapsid_{cfg.get('name', 'default')}",
        hash_store_dir=hash_store_dir,
        logger_name="kaptive",
    )

    orch.run(
        inputs=genome_files,
        output_func=bound_output,
        worker=bound_worker,
        code_id=code_version,
        parallelism=run_cfg.n_jobs,
        force=force,
        tqdm_desc=f"kaptive processing ({cfg.get('name', 'default')})",
    )

    # 6. Manifest generation
    kaptive_files = list(output_dir.glob("*.json"))
    manifest_file = Path(str(output_dir) + "_manifest.json")
    create_manifest(kaptive_files, manifest_file)
    logger.info(f"✅ Kaptive workflow complete. Manifest: {manifest_file}")


def main():
    parser = argparse.ArgumentParser(description="Run K-types from kaptive3.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument(
        "--code_hash", action="append", type=Path, required=True, help="Code hash file(s)"
    )
    parser.add_argument("--hash_store", type=Path, help="Workflow hash storage directory")
    parser.add_argument("--force", action="store_true", help="Force rerun")

    args = parser.parse_args()

    run_kaptive_workflow(
        config_path=args.config,
        code_hash_files=args.code_hash,
        hash_store=args.hash_store,
        force=args.force,
    )


if __name__ == "__main__":
    main()
