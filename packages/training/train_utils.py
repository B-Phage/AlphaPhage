import os

import h5py
import numpy as np
import utils.consol_log as log
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, IterableDataset

logger = log.get_logger(level=log.WARNING)


def get_token_basics(features_path, dataset_name_host, dataset_name_phage):
    """Get length_padded, num_tokens, tokens_in_data from the HDF5 features
    file.

    Args:
        features_path : (str) Path to HDF5 file with features.
        dataset_name_host : (str) Name of dataset in HDF5 file for host data.
        dataset_name_phage : (str) Name of dataset in HDF5 file for phage data.
    Returns:
        length_padded : (int) Length of concatenated host-phage sequences.
        num_tokens : (int) Number of unique tokens in the data.
        tokens_in_data : (list) List of unique tokens in the data.
    """
    with h5py.File(features_path, "r", libver="latest", swmr=True) as f_x:
        Xh5_host = f_x[dataset_name_host]
        Xh5_phage = f_x[dataset_name_phage]

        # Just getting length_padded as length of first pair joined
        length_padded = len(Xh5_host[0]) + len(Xh5_phage[0])
        logger.info(f"length_padded  : {length_padded}")

        tokens_in_data = set({})
        for i in range(len(Xh5_host)):
            tokens_in_data.update(set(Xh5_host[i].astype(np.int32)))
        for i in range(len(Xh5_phage)):
            tokens_in_data.update(set(Xh5_phage[i].astype(np.int32)))
        tokens_in_data = list(tokens_in_data)
        num_tokens = max(tokens_in_data) + 1
        logger.info(f"num_tokens  : {num_tokens}")

    return length_padded, num_tokens, tokens_in_data


def get_experiment_tracker_method(cfg: DictConfig, experiment_tracker_dir=str):
    if cfg.trainer.experiment_tracker_method == "DVCLiveLogger":
        from pathlib import Path

        from dvclive.lightning import DVCLiveLogger

        experiment_tracker = DVCLiveLogger(
            dir=experiment_tracker_dir,
            log_model=False,  # Optionally log model checkpoints as DVC artifacts
            dvcyaml=False,  # Optionally track params/metrics in dvc.yaml
        )
        return experiment_tracker
    else:
        raise ValueError(f"Unknown logger method: {cfg.trainer.experiment_tracker_method}")


def get_datamodule(cfg: DictConfig) -> IterableDataset:
    # Only BinaryPairkeyDataModule is implemented.
    if cfg.reader_mode == "binary_pairkey":
        from dataloaders.dataloaders import BinaryPairkeyDataModule

        return BinaryPairkeyDataModule(cfg)
    else:
        raise ValueError(f"Unknown reader_mode: {cfg.reader_mode}")


def get_model(cfg: DictConfig, num_tokens: int, vocab_size_out: int, length_padded: int):
    # Only TransformerX is implemented.
    if cfg.architecture == "TransformerX":
        from arc.TransformerX import TransformerX, TransformerXClassifier, archspecs, archspecs2

        return TransformerX(cfg, num_tokens, vocab_size_out, length_padded, archspecs, archspecs2)
    else:
        raise ValueError(f"Unknown architecture: {cfg.architecture}")
