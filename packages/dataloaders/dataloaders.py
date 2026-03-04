import random
import sys
from operator import index

import h5py
import numpy as np
import pytorch_lightning as pl
import torch
import utils.consol_log as log
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Sampler

logger = log.get_logger(level=log.ERROR)


# Runs at the beginning of each batch
class IndexBatchSampler(Sampler):
    """A generator to supply batch indices to accompany a DataLoader.

    Note: Do not use batch_size, shuffle, or sampler in the DataLoader, but
    supply this BatchSampler object as value to the batch_sampler argument in
    the DataLoader object with the full dataset.
    """

    def __init__(self, indices, batch_size):
        """
        index_groups: List of lists, e.g., [[0, 5, 10], [1, 6, 11], ...]
        Each inner list represents the indices for one batch.

        Args:
            index -- (list) List of relevant indices.
            batch_size : (int|'all') The size of each batch.
        Returns:
            index_groups : numpy array of sub-lists, where each sub-list
            represents the indices mapping to a batch.

        """

        self.indices = indices
        self.batch_size = batch_size
        self.num_samples = len(self.indices)
        self.index_shuf = self.indices.copy()
        if isinstance(self.batch_size, int):
            # Would incorporate residual
            self.num_batches = len(self.index_shuf) // self.batch_size + 1
        elif self.batch_size == "all":
            # Take all indices
            self.batch_size = len(self.index_shuf)
            self.num_batches = 1
        else:
            raise ValueError("batch_size not type int or 'all'. ")
        random.shuffle(self.index_shuf)
        self.make_index_groups()

    def make_index_groups(self):
        """Create a lists of index groups based on the current shuffled indices."""
        self.index_groups = np.array_split(self.index_shuf, self.num_batches)

    def permutate(self):
        """Permutate the indices and recreate the index groups."""
        random.shuffle(self.index_shuf)
        self.make_index_groups()

    def __iter__(self):
        self.permutate()
        for batch_indices in self.index_groups:
            yield batch_indices

    def __len__(self):
        # number of batches
        return len(self.index_groups)


class BinaryPairkeyFeatureDataset(Dataset):
    """Data reader for accessing host and phage features separately from keys in
    h5 format and pairing them.

    Args:
        cfg:
        features_path :
        labels_path :
    Returns:
        torch Dataset object with __getitem__ and __len__ methods. The
        __getitem__ method returns:
            A tuple (X, y) where:
            X : torch.LongTensor of concatenated host and phage features.
            y : torch.LongTensor of the label.
        The __len__ method returns:
            The total number of samples in the dataset.
    Yields:
        An iterator in which the elements are a side-by-side concatenation of
        host and phage features.
    Example:
        pair_indices_name   = 'pair_indices'
        dataset_name_host   = 'host_vec_1'
        dataset_name_phage  = 'phage_vec_1'
    """

    def __init__(self, cfg, features_path, labels_path):

        # If true, replaces h5 read for dict read for unit testing. Overridden
        # to True by Mocking in unit tests.

        self.features_path = features_path
        self.labels_path = labels_path

        self.pair_indices_name = cfg.pair_indices_name
        self.dataset_name_host = cfg.dataset_name_host
        self.dataset_name_phage = cfg.dataset_name_phage
        self.batch_size = cfg.batch_size

        # Pre-calculate length
        with h5py.File(self.labels_path, "r") as f_y:
            self.total_size = len(f_y["data"])

        ## workers must == 0
        self.open_h5()

        # Map the datasets to the persistent file handles. workers must == 0
        self.Ynp = self.f_y["data"]
        self.Xh5_host = self.f_x[self.dataset_name_host]
        self.Xh5_phage = self.f_x[self.dataset_name_phage]
        self.pair_indices = self.f_x[self.pair_indices_name]

    def open_h5(self):
        """Reading an h5 file"""
        if not hasattr(self, "Ynp"):  # Lazy loading avoids opening an already opened file
            self.f_y = h5py.File(self.labels_path, "r", libver="latest", swmr=True)
            self.f_x = h5py.File(self.features_path, "r", libver="latest", swmr=True)

    def __getitem__(self, idx):
        """Returns a single sample consisting of concatenated host and phage
        features, and the label.
        """

        ii_host, ii_phage = self.pair_indices[idx]
        data_host = self.Xh5_host[ii_host]
        data_phage = self.Xh5_phage[ii_phage]

        return (
            torch.from_numpy(np.concatenate((data_host, data_phage), axis=0)).long(),
            torch.tensor(self.Ynp[idx]).long(),
        )

    def __len__(self):
        return self.total_size


class BinaryPairkeyDataModule(pl.LightningDataModule):
    """A LightningDataModule for loading host-phage data in the
    BinaryPairkeyFeatureDataset format, accessing host and phage features
    separately from keys in h5 format and pairing them.

    Args:
        argument -- description
    Return: return_description
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.batch_size = cfg.batch_size
        # self.save_hyperparameters()

    def setup(self, stage=None):
        if self.cfg.debug_mode:
            features_path = self.cfg.debug_features_path
            labels_path = self.cfg.debug_labels_path
        else:
            features_path = self.cfg.train_features_path
            labels_path = self.cfg.train_labels_path

        # Build out stages: "fit", "validate", "test", or "predict"
        if stage == "fit" or stage is None:
            self.full_dataset = BinaryPairkeyFeatureDataset(self.cfg, features_path, labels_path)

            all_indices = list(range(len(self.full_dataset)))
            # Partition indices
            self.train_idx, self.val_idx = train_test_split(
                all_indices, test_size=self.cfg.val_fraction, random_state=self.cfg.seed
            )
            print("Setup: Train and Val sets created.")

        if stage == "test" or stage is None:
            self.test_dataset = BinaryPairkeyFeatureDataset(self.cfg, features_path, labels_path)
            print("Setup: Test set created.")

        # 3. Logic for Production Prediction
        if stage == "predict":
            self.predict_dataset = BinaryPairkeyFeatureDataset(self.cfg, features_path, labels_path)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.full_dataset,
            batch_sampler=IndexBatchSampler(
                self.train_idx, self.batch_size
            ),  # This passes index lists to dataset.__getitem__. Not compatible with shuffle or sampler args
            num_workers=self.cfg.num_workers,  # Parallel loading
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.full_dataset,
            batch_sampler=IndexBatchSampler(self.val_idx, self.batch_size),  # Pass val indices only
            num_workers=self.cfg.num_workers,
        )
