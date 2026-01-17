import os
import sys

import h5py
import hydra
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import utils.consol_log as log
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader, IterableDataset
from training.train_utils import get_token_basics

logger = log.get_logger(level=log.ERROR)


config_path = "../../tests/unit/data"
config_name = "config_test_2"


@hydra.main(config_path=config_path, config_name=config_name, version_base="1.3")
def main(cfg: DictConfig):
    # ----------------------------------------------------
    # A. Setup Logging and Reproducibility
    # ----------------------------------------------------
    logger = log.get_logger(level=cfg.verboselevel)

    logger.info(f"\n{OmegaConf.to_yaml(cfg)}")  # Print final merged config

    if cfg.seed is not None:
        # For testing purposes, set seed for reproducibility
        pl.seed_everything(cfg.seed, workers=True)

    # Set the logging directory based on Hydra's output
    log_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    logger.warning(f"Writing logs at {log_dir}")
    os.makedirs(os.path.join(log_dir, "checkpoints"), exist_ok=True)

    # ----------------------------------------------------
    # B. Define Callbacks for Checkpointing and Early Stopping
    # ----------------------------------------------------
    # 1. ModelCheckpoint: Saves the best model based on validation loss
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(log_dir, "checkpoints"),
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
    vocab_size_out = cfg.hyp.vocab_size_out
    length_padded, num_tokens, tokens_in_data = get_token_basics(
        cfg.train_features_path, cfg.dataset_name_host, cfg.dataset_name_phage
    )

    # Only BinaryPairkeyDataModule is implemented.
    if cfg.reader_mode == "binary_pairkey":
        from dataloaders.dataloaders import BinaryPairkeyDataModule

        datamodule = BinaryPairkeyDataModule(cfg)
    else:
        raise ValueError(f"Unknown reader_mode: {cfg.reader_mode}")

    # Only TransformerX is implemented.
    if cfg.architecture == "TransformerX":
        from arc.TransformerX import TransformerX, TransformerXClassifier, archspecs, archspecs2

        model = TransformerX(cfg, num_tokens, vocab_size_out, length_padded, archspecs, archspecs2)
    else:
        raise ValueError(f"Unknown architecture: {cfg.architecture}")

    # ----------------------------------------------------
    # D. Initialize Trainer (The Engineering Code)
    # ----------------------------------------------------

    if cfg.trainer.logger_method == "CSVLogger":
        logger_object = pl.loggers.CSVLogger(log_dir)

    elif cfg.trainer.logger_method == "WandbLogger":
        logger_object = pl.loggers.WandbLogger(
            project=cfg.trainer.project,
            name=cfg.trainer.run_name,
            save_dir=log_dir,
        )

    elif cfg.trainer.logger_method == "TensorBoardLogger":
        logger_object = pl.loggers.TensorBoardLogger(log_dir, name=cfg.trainer.run_name)

    elif cfg.trainer.logger_method == "DVCLiveLogger":
        from pathlib import Path

        from dvclive.lightning import DVCLiveLogger

        logger_object = DVCLiveLogger(
            dir=Path(os.path.join(log_dir, "mlrun")),
            log_model=False,  # Optionally log model checkpoints as DVC artifacts
        )

    elif cfg.trainer.logger_method == "MLFlowLogger":
        if cfg.trainer.mlflow_tracking_uri is not None:
            # Requires a localhost e.g. http://localhost:5000 or
            # a remote server e.g. http://<IP_ADDRESS>:5000
            logger_object = pl.loggers.MLFlowLogger(
                experiment_name=cfg.trainer.run_name, tracking_uri=cfg.trainer.mlflow_tracking_uri
            )
        else:
            from pathlib import Path

            # writes to local file system inside hydra logging directory
            logger_object = pl.loggers.MLFlowLogger(
                experiment_name=cfg.trainer.run_name,
                tracking_uri=Path(os.path.join(log_dir, "mlrun")).as_uri(),
            )

    else:
        raise ValueError(f"Unknown logger method: {cfg.trainer.logger_method}")

    trainer = pl.Trainer(
        max_epochs=cfg.trainer.epochs,
        accelerator=cfg.trainer.accelerator,  # e.g., 'gpu', 'cpu'
        devices=cfg.trainer.devices,  # e.g., 1 or [0, 1]
        logger=logger_object,
        callbacks=[checkpoint_callback, early_stop_callback],
        # Precision setting is important for modern hardware
        precision=cfg.trainer.precision,
        # strategy="ddp" #for devices >1 and DistributedDataParallel
        enable_progress_bar=True,
        profiler="advanced",
        # check_val_every_n_epoch=1,       # Run validation every N epochs
    )
    # # example using built in slurm plugin
    # from lightning.fabric.plugins.environments import SLURMEnvironment
    # trainer = Trainer(plugins=[SLURMEnvironment()])

    # ----------------------------------------------------
    # E. Train the Model
    # ----------------------------------------------------
    logger.warning(f"\nStarting training...")
    trainer.fit(model, datamodule=datamodule)

    # Optionally, load and save the best checkpoint
    best_path = checkpoint_callback.best_model_path
    logger.warning(f"\nTraining finished. Best model saved at: {best_path}")


if __name__ == "__main__":
    main()
