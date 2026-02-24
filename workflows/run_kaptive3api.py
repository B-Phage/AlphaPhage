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
from kaptive.assembly import typing_pipeline
from kaptive.database import load_database
from omegaconf import DictConfig, OmegaConf
from orchestration.orchestrator import Orchestrator
from utils.config_loader import get_config
from utils.hash import compute_combined_code_version, create_manifest

logger = log.get_logger(level=log.ERROR)


# def get_output_dir(cfg):
#     return cfg.kaptive_dir


# -----------------------------
# Output function
# -----------------------------
def kapsid_output_func(input_path: Path, output_dir: Path):
    """Provides the expected output file path(s) for a given input file. The
    Orchestrator uses this to determine where to look for outputs and whether
    they already exist.
    """
    # output_dir = get_output_dir(cfg)
    genome_name = Path(input_path).stem
    kaptive_file = os.path.join(output_dir, f"{genome_name}.json")

    return [Path(kaptive_file)]


def run_kapsid(input_path: Path, outputs: list, db_obj: Any):
    """Implements the kapsid.typing_pipeline. The Orchestrator expects a function
    that takes an input path and a list of output paths, and returns the list of
    output paths after processing. The db_obj is passed in via
    Note: The Orchestrator worker requires two arguments - input_path and
    outputs. The db_obj is passed in via functools.partial in main to create a
    bound version of this function that only requires input_path and outputs.
    """
    kaptive_file = outputs[0]
    result = typing_pipeline(input_path, db_obj)
    d = {}
    if result:
        # Request the JSON string
        json_output = result.format("json")
        if isinstance(json_output, str):
            dresult = json.loads(json_output)
        else:
            # Handle cases where format() didn't return a string as expected
            raise TypeError(f"Expected JSON string from format('json'), got {type(json_output)}")

        for key in [
            "sample_name",
            "best_match",
            "confidence",
            "phenotype",
            "problems",
            "percent_identity",
            "percent_coverage",
        ]:
            d.update({key: dresult[key]})

    with open(kaptive_file, "w") as h:
        json.dump(d, h)

    return result


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
