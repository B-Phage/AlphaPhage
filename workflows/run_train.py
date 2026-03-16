#!/usr/bin/env python3
"""
Run Trainer.
"""

import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="torch.cuda")
# This targets the specific deprecation message about treespec/LeafSpec
warnings.filterwarnings("ignore", message=r".*isinstance\(treespec, LeafSpec\).*")
# If you want to be extra sure to catch all library-level deprecation warnings:
warnings.filterwarnings("ignore", category=DeprecationWarning)

import argparse
import os
import sys
from pathlib import Path
from typing import cast

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import utils.consol_log as log
from omegaconf import DictConfig, OmegaConf
from orchestration.orchestrator import Orchestrator
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from training.train_utils import (
    check_model_size,
    get_datamodule,
    get_experiment_tracker_method,
    get_model,
    get_token_basics,
)
from utils.config_loader import get_config_generic
from utils.hash import compute_combined_code_version, create_manifest

logger = log.get_logger(level=log.ERROR)


# -----------------------------
# Load configuration
# -----------------------------
def get_output_dir(cfg):
    EXP_TRACK_DIR = cfg.experiment_tracker_dir
    RUN_NAME = cfg.get("run_name", "default_run")
    return os.path.join(EXP_TRACK_DIR, RUN_NAME)


def run_trainer(input_path: Path, outputs: list):
    # ----------------------------------------------------
    # A. Setup Logging and Reproducibility
    # ----------------------------------------------------
    cfg = cast(DictConfig, OmegaConf.load(input_path))

    output_dir = get_output_dir(cfg)

    logger = log.get_logger(level=cfg.verboselevel)

    logger.info(f"\n{OmegaConf.to_yaml(cfg)}")  # Print final merged config

    if cfg.seed is not None:
        pl.seed_everything(cfg.seed, workers=True)

    logger.warning(f"Writing logs at {output_dir}")

    # ----------------------------------------------------
    # B. Define Callbacks for Checkpointing and Early Stopping
    # ----------------------------------------------------
    # 1. ModelCheckpoint: Saves the model checkpoint with the best validation loss(ses)
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(output_dir, "checkpoints"),
        filename="best-model-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=cfg.save_top_k,
        verbose=True,
    )

    # 2. EarlyStopping: Stops training if val_loss doesn't improve
    early_stop_callback = EarlyStopping(
        monitor="val_loss", patience=cfg.patience, mode="min", verbose=True
    )

    # ----------------------------------------------------
    # C. Initialize Components
    # ----------------------------------------------------
    length_padded, num_tokens, tokens_in_data = get_token_basics(
        cfg.train_features_path, cfg.dataset_name_host, cfg.dataset_name_phage
    )
    datamodule = get_datamodule(cfg)
    datamodule.setup(stage="fit")  # Required to define train_idx
    weights = datamodule.get_class_weights()

    model = get_model(cfg, num_tokens, cfg.vocab_size_out, length_padded, class_weights=weights)
    check_model_size(model)  # Check for data leaks

    experiment_tracker_dir = os.path.join(output_dir, "tracker")
    experiment_tracker = get_experiment_tracker_method(cfg, experiment_tracker_dir)

    # ----------------------------------------------------
    # D. Initialize Trainer (The Engineering Code)
    # ----------------------------------------------------
    trainer = pl.Trainer(
        max_epochs=cfg.epochs,
        accelerator=cfg.accelerator,  # e.g., 'gpu', 'cpu'
        devices=cfg.devices,  # e.g., 1 or [0, 1]
        logger=experiment_tracker,
        callbacks=[checkpoint_callback, early_stop_callback],
        precision=cfg.precision,
        enable_progress_bar=True,
        profiler="advanced" if cfg.get("debug_mode", False) else None,
    )

    # ----------------------------------------------------
    # E. Train the Model
    # ----------------------------------------------------
    logger.warning(f"\nStarting training...")
    trainer.fit(model, datamodule=datamodule)

    # Optionally, load and save the best checkpoint
    best_path = checkpoint_callback.best_model_path
    logger.warning(f"\nTraining finished. Best model saved at: {best_path}")

    return outputs


# -----------------------------
# Output function
# -----------------------------
def trainer_output_func(input_path: Path):
    cfg = OmegaConf.load(input_path)
    output_dir = get_output_dir(cfg)
    training_run_name = Path(input_path).stem
    training_run_name_file = os.path.join(output_dir, f"{training_run_name}.csv")
    return [Path(training_run_name_file)]


# -----------------------------
# Run orchestrator
# -----------------------------
if __name__ == "__main__":

    # from utils.config_loader import get_train_config

    config = "params.yaml"
    force = False

    # -----------------------------
    # Command-line args
    # -----------------------------

    parser = argparse.ArgumentParser(description="Run Trainer")
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
        # type=bool,
        help="Force rerun all genomes regardless of hash",
    )
    args = parser.parse_args()
    if args.config:
        config = args.config
    if args.force and args.force == True:
        force = True

    # -----------------------------
    # Prepare inputs
    # -----------------------------
    cfg = cast(DictConfig, get_config_generic(path=args.config))

    run_cfg = cast(
        DictConfig, get_config_generic(path="ml_assets/config/runtime.yaml", params_set="meta")
    )

    EXP_TRACK_DIR = cfg.experiment_tracker_dir
    RUN_NAME = cfg.get("run_name", "default_run")
    N_JOBS = run_cfg.n_jobs
    CONF_CODE_VERSION = cfg.code_version
    HASH_STORE_DIR = Path("ml_assets/data/workflow/hashes")

    lst_training_inputs = [Path(args.config)]

    # output_dir = os.path.join(EXP_TRACK_DIR, RUN_NAME)
    output_dir = get_output_dir(cfg)

    # Note: output_dir->os.path.join(OUTPUT_DIR, RUN_NAME, "checkpoints")

    CODE_VERSION = compute_combined_code_version(
        base_version=CONF_CODE_VERSION,
        hash_files=args.code_hash,
        config_items=[cfg],
    )

    os.makedirs(output_dir, exist_ok=True)
    orch = Orchestrator(
        workflow_name=f"train_{cfg.get('run_name', 'default')}",
        hash_store_dir=HASH_STORE_DIR,
        logger_name="train",
    )
    orch.run(
        inputs=lst_training_inputs,
        output_func=trainer_output_func,
        worker=run_trainer,
        code_id=CODE_VERSION,
        parallelism=N_JOBS,
        force=args.force,
        tqdm_desc=f"Prodigal processing ({cfg.get('run_name', 'default')})",
    )

    trainer_files = list(Path(output_dir).glob("*.json"))
    MANIFEST_FILE = Path(cfg.experiment_tracker_dir + "_manifest.json")
    create_manifest(trainer_files, MANIFEST_FILE)
