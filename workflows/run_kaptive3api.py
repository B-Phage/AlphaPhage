#!/usr/bin/env python3
"""
Run K-types from kaptive3.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import pandas as pd
import utils.consol_log as log
from kaptive.assembly import typing_pipeline
from kaptive.database import load_database
from omegaconf import DictConfig, OmegaConf
from orchestration.orchestrator import Orchestrator
from utils.config_loader import get_kapsid_config

logger = log.get_logger(level=log.ERROR)


def get_output_dir(cfg):
    EXP_TRACK_DIR = cfg.trainer.experiment_tracker_dir
    RUN_NAME = cfg.trainer.get("run_name", "default_run")
    return os.path.join(EXP_TRACK_DIR, RUN_NAME)


def run_kapsid(input_path: str, outputs: list):

    cfg = OmegaConf.load(input_path)
    output_dir = get_output_dir(cfg)

    logger = log.get_logger(level=cfg.verboselevel)

    experiment_tracker_dir = os.path.join(output_dir, "tracker")

    logger.warning(f"\nStarting kapsid...")

    outputs = typing_pipeline(input_path, db)  # NILS37 NILS37

    logger.warning(f"\Kapsid finished.")

    return outputs


# -----------------------------
# Output function
# -----------------------------
def kapsid_output_func(input_path: Path):
    cfg = OmegaConf.load(input_path)
    output_dir = get_output_dir(cfg)
    genome_name = Path(input_path).stem
    genome_name_file = os.path.join(output_dir, f"{genome_name}.csv")
    return [Path(genome_name_file)]


if __name__ == "__main__":

    logger.info("Doing arg parsing for msa.py")
    parser = argparse.ArgumentParser()
    parser.add_argument("-dd", "--data_dir", nargs=1, type=str)
    parser.add_argument("-fd", "--fastadir", nargs=1, type=str)
    parser.add_argument("-od", "--outdir", nargs=1, type=str)
    parser.add_argument("-on", "--orgdb_name", nargs=1, type=str)
    args = parser.parse_args()

    if args.data_dir == None or args.data_dir[0] == "None":
        logger.error(f"No data_dir. Exiting.")
        sys.exit()
    else:
        data_dir = args.data_dir[0]

    if args.fastadir == None or args.fastadir[0] == "None":
        logger.error(f"No fastadir. Exiting.")
        sys.exit()
    else:
        fastadir = args.fastadir[0]

    if args.outdir == None or args.outdir[0] == "None":
        logger.error(f"No outdir. Exiting.")
        sys.exit()
    else:
        outdir = args.outdir[0]

    if args.orgdb_name == None or args.orgdb_name[0] == "None":
        logger.error(f"No orgdb_name. Exiting.")
        sys.exit()
    else:
        orgdb_name = args.orgdb_name[0]

    # max_rows = None  # for testing
    # refers to the db name e.g. 'ecoli_k' in _DB_KEYWORDS = {..'EC-K-typing_group2and3_v1.3': ['ecoli_k']} in database.py
    # at /Users/<username>/miniforge3/envs/<conda env>/lib/python<ver>/site-packages/kaptive/database.py

    # data_dir = os.path.abspath('../databases')
    # fastadir = os.path.join(data_dir, 'Gaborieau_Picard/Picard_Ecoli_25941691')
    # outdir = os.path.abspath('proc/ann_host/out/kaptive')
    # orgdb_name = 'ecoli_k'
    db = load_database(
        orgdb_name
    )  # Load the Ecoli K locus database once and pass it to the typing pipeline

    # if not os.path.exists(outdir):
    #     os.mkdir(outdir)

    cfg_path, cfg = get_kapsid_config(override_path=args.config)

    # -----------------------------
    # Prepare inputs
    # -----------------------------
    lst_kapsid_inputs = [cfg_path]

    EXP_TRACK_DIR = cfg.output_dir
    RUN_NAME = cfg.get("run_name", "default_run")

    # output_dir = os.path.join(EXP_TRACK_DIR, RUN_NAME)
    output_dir = get_output_dir(cfg)

    # Note: output_dir->os.path.join(OUTPUT_DIR, RUN_NAME, "checkpoints")
    N_JOBS = cfg.num_workers
    CODE_VERSION = cfg.code_version
    HASH_STORE_DIR = Path("ml_assets/data/workflow/hashes")

    os.makedirs(output_dir, exist_ok=True)
    orch = Orchestrator(
        workflow_name=f"kapsid_{cfg.get('name', 'default')}",
        hash_store_dir=HASH_STORE_DIR,
        logger_name="kapsid",
    )
    orch.run(
        inputs=lst_kapsid_inputs,
        output_func=kapsid_output_func,
        worker=run_kapsid,
        code_id=CODE_VERSION,
        parallelism=N_JOBS,
        force=args.force,
        tqdm_desc=f"Kapsid processing ({cfg.get('name', 'default')})",
    )
