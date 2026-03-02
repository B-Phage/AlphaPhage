#!/usr/bin/env python3
"""
Run K-types from kaptive3.
"""

import argparse
import json
import logging
import os
import sys
from functools import partial
from pathlib import Path
from typing import Any

import pandas as pd
import utils.consol_log as log
from kaptive3api.kaptive3api_functions import kapsid_output_func, run_kapsid
from kaptive.assembly import typing_pipeline
from kaptive.database import load_database
from omegaconf import DictConfig, OmegaConf
from orchestration.orchestrator import Orchestrator
from utils.config_loader import get_config
from utils.hash import compute_combined_code_version, create_manifest

logger = log.get_logger(level=log.ERROR)


if __name__ == "__main__":

    logger.info("Doing arg parsing for msa.py")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=False,
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
        help="Force rerun all genomes regardless of hash",
    )
    args = parser.parse_args()
    if args.config:
        config = args.config
    if args.force and args.force == True:
        force = True

    # -----------------------------
    # Load configuration
    # -----------------------------
    run_cfg = get_config(
        "ml_assets/config/runtime.yaml", params_set="meta"
    )  # config that does not affect outputs
    gen_cfg = get_config(override_path=args.config, params_set="general")

    cfg = get_config(override_path=args.config, params_set="kaptive_config")

    LST_GENOMES = gen_cfg.lst_genomes
    N_JOBS = run_cfg.n_jobs
    CONF_CODE_VERSION = gen_cfg.code_version

    CODE_VERSION = compute_combined_code_version(
        base_version=CONF_CODE_VERSION,
        hash_files=args.code_hash,
        config_items=[cfg],
    )

    GENOME_DIR = Path(cfg.genome_dir)
    KAPTIVE_DIR = Path(cfg.kaptive_dir)
    HASH_STORE_DIR = Path("ml_assets/data/workflow/hashes")

    # -----------------------------
    # Prepare inputs
    # -----------------------------
    if LST_GENOMES:
        genome_files = [GENOME_DIR / f"{g}.fasta" for g in LST_GENOMES]
    else:
        genome_files = sorted(GENOME_DIR.glob("*.fasta"))

    orgdb_full_path = cfg.orgdb_full_path
    db = load_database(orgdb_full_path)

    # 3. Create the "Orchestrator-compatible" version
    # This creates a 'callable' that only asks for input_path and outputs
    # because db_obj is already satisfied.
    output_dir = Path(cfg.kaptive_dir)
    bound_worker = partial(run_kapsid, db_obj=db)
    bound_output = partial(kapsid_output_func, output_dir=output_dir)

    # output_dir = get_output_dir(cfg)
    os.makedirs(output_dir, exist_ok=True)

    orch = Orchestrator(
        workflow_name=f"kapsid_{cfg.get('name', 'default')}",
        hash_store_dir=HASH_STORE_DIR,
        logger_name="kaptive",
    )
    orch.run(
        inputs=genome_files,
        output_func=bound_output,
        worker=bound_worker,
        code_id=CODE_VERSION,
        parallelism=N_JOBS,
        force=args.force,
        tqdm_desc=f"kaptive processing ({cfg.get('name', 'default')})",
    )

    kaptive_files = list(Path(output_dir).glob("*.json"))
    MANIFEST_FILE = Path(cfg.kaptive_dir + "_manifest.json")
    create_manifest(kaptive_files, MANIFEST_FILE)
