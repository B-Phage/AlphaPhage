import os

import h5py
import numpy as np
import utils.consol_log as log
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, IterableDataset

logger = log.get_logger()


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

        assert isinstance(Xh5_host, h5py.Dataset)
        assert isinstance(Xh5_phage, h5py.Dataset)

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


def get_experiment_tracker_method(cfg: DictConfig, experiment_tracker_dir: str):
    if cfg.experiment_tracker_method == "DVCLiveLogger":
        from pathlib import Path

        from dvclive.lightning import DVCLiveLogger

        experiment_tracker = DVCLiveLogger(
            dir=experiment_tracker_dir,
            log_model=False,  # Optionally log model checkpoints as DVC artifacts
            dvcyaml=False,  # Optionally track params/metrics in dvc.yaml
            report="html",  # This generates the 'report.html' automatically
        )
        return experiment_tracker
    else:
        raise ValueError(f"Unknown logger method: {cfg.experiment_tracker_method}")


def get_datamodule(cfg: DictConfig):
    # Only BinaryPairkeyDataModule is implemented.
    if cfg.reader_mode == "binary_pairkey":
        from dataloaders.dataloaders import BinaryPairkeyDataModule

        return BinaryPairkeyDataModule(cfg)
    else:
        raise ValueError(f"Unknown reader_mode: {cfg.reader_mode}")


def get_model(
    cfg: DictConfig, num_tokens: int, vocab_size_out: int, length_padded: int, class_weights=None
):
    # Only TransformerX is implemented.
    if cfg.architecture == "TransformerX":
        from models.transformer_archspecs import archspecs, archspecs2
        from models.transformer_module import TransformerX

        return TransformerX(
            cfg,
            num_tokens,
            vocab_size_out,
            length_padded,
            archspecs,
            archspecs2,
            class_weights=class_weights,
        )
    else:
        raise ValueError(f"Unknown architecture: {cfg.architecture}")


def check_model_size(model):
    from sys import getsizeof

    logger.warning(f"{'Name':<50} | {'Size (MB)':<10}")
    logger.warning("-" * 65)

    # Check Parameters
    for name, param in model.named_parameters():
        size = param.element_size() * param.nelement() / 1e6
        logger.warning(f"{name:<50} | {size:>10.2f}")

    # Check Buffers (The likely source of the 300MB)
    for name, buffer in model.named_buffers():
        size = buffer.element_size() * buffer.nelement() / 1e6
        logger.warning(f"{name:<50} | {size:>10.2f}")
