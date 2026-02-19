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

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import utils.consol_log as log
from omegaconf import DictConfig, OmegaConf
from orchestration.orchestrator import Orchestrator
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from training.train_utils import (
    get_datamodule,
    get_experiment_tracker_method,
    get_model,
    get_token_basics,
)

logger = log.get_logger(level=log.ERROR)

# -----------------------------
# Load configuration
# -----------------------------


def get_output_dir(cfg):
    EXP_TRACK_DIR = cfg.trainer.experiment_tracker_dir
    RUN_NAME = cfg.trainer.get("run_name", "default_run")
    return os.path.join(EXP_TRACK_DIR, RUN_NAME)


def run_trainer(input_path: Path, outputs: list):
    # ----------------------------------------------------
    # A. Setup Logging and Reproducibility
    # ----------------------------------------------------
    cfg = OmegaConf.load(input_path)
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
        save_top_k=cfg.trainer.save_top_k,
        verbose=True,
    )

    # 2. EarlyStopping: Stops training if val_loss doesn't improve
    early_stop_callback = EarlyStopping(
        monitor="val_loss", patience=cfg.trainer.patience, mode="min", verbose=True
    )

    # ----------------------------------------------------
    # C. Initialize Components
    # ----------------------------------------------------
    length_padded, num_tokens, tokens_in_data = get_token_basics(
        cfg.train_features_path, cfg.dataset_name_host, cfg.dataset_name_phage
    )

    datamodule = get_datamodule(cfg)

    model = get_model(cfg, num_tokens, cfg.hyp.vocab_size_out, length_padded)

    experiment_tracker_dir = os.path.join(output_dir, "tracker")
    experiment_tracker = get_experiment_tracker_method(cfg, experiment_tracker_dir)

    # ----------------------------------------------------
    # D. Initialize Trainer (The Engineering Code)
    # ----------------------------------------------------
    trainer = pl.Trainer(
        max_epochs=cfg.trainer.epochs,
        accelerator=cfg.trainer.accelerator,  # e.g., 'gpu', 'cpu'
        devices=cfg.trainer.devices,  # e.g., 1 or [0, 1]
        logger=experiment_tracker,
        callbacks=[checkpoint_callback, early_stop_callback],
        precision=cfg.trainer.precision,
        enable_progress_bar=True,
        profiler="advanced",
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

    from utils.config_loader import get_train_config

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

    cfg_path, cfg = get_train_config(override_path=args.config)

    # -----------------------------
    # Prepare inputs
    # -----------------------------
    lst_training_inputs = [cfg_path]

    EXP_TRACK_DIR = cfg.trainer.experiment_tracker_dir
    RUN_NAME = cfg.trainer.get("run_name", "default_run")

    # output_dir = os.path.join(EXP_TRACK_DIR, RUN_NAME)
    output_dir = get_output_dir(cfg)

    # Note: output_dir->os.path.join(OUTPUT_DIR, RUN_NAME, "checkpoints")
    N_JOBS = cfg.num_workers
    CODE_VERSION = cfg.code_version
    HASH_STORE_DIR = Path("ml_assets/data/workflow/hashes")

    os.makedirs(output_dir, exist_ok=True)
    orch = Orchestrator(
        workflow_name=f"train_{cfg.get('name', 'default')}",
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
        tqdm_desc=f"Prodigal processing ({cfg.get('name', 'default')})",
    )
