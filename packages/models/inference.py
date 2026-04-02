from pathlib import Path

import h5py
import numpy as np
import torch
from dataloaders.dataloaders import BinaryPairkeyFeatureDataset
from models.transformer_module import TransformerX
from omegaconf import DictConfig
from torch.utils.data import DataLoader


class TransformerInferencer:
    def __init__(self, checkpoint_path: str, cfg: DictConfig):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # We still need these dicts to rebuild the architecture layers
        from models.transformer_archspecs import archspecs, archspecs2

        # 1. Load Model
        # Lightning automatically pulls num_tokens, length_padded, etc.,
        # from the 'hyper_parameters' we just fixed.
        self.model = TransformerX.load_from_checkpoint(
            checkpoint_path,
            archspecs=archspecs,
            archspecs2=archspecs2,
            map_location=self.device,
            strict=False,
        )
        self.model.to(self.device)
        self.model.eval()

        # 2. Extract "The Truth" from the model itself
        self.expected_len = self.model.hparams["length_padded"]
        self.expected_vocab = self.model.hparams["num_tokens"]

        print(
            f"✅ Model Loaded. Expected Sequence Length: {self.expected_len}, Max Token ID: {self.expected_vocab - 1}"
        )

    def validate_hdf5(self, features_path: str):
        """Pre-inference check to ensure the H5 file matches the model weights."""

        # 1. Check Dimensions (Width)
        with h5py.File(features_path, "r") as f:
            # Shape is (N_samples, Width)
            h_width = f[self.model.cfg.dataset_name_host].shape[1]
            p_width = f[self.model.cfg.dataset_name_phage].shape[1]
            obs_width = h_width + p_width

            # Also get max token for the vocabulary check
            # We look for the maximum value stored in the integer datasets
            h_max = f[self.model.cfg.dataset_name_host][:].max()
            p_max = f[self.model.cfg.dataset_name_phage][:].max()
            obs_tokens = max(h_max, p_max) + 1  # +1 to compare against vocab size

        # 2. Logic: Sequence Length (Positional Mismatch)
        if obs_width != self.expected_len:
            raise ValueError(
                f"📏 DIMENSION MISMATCH: Model expects {self.expected_len} total features, "
                f"but HDF5 has {obs_width} ({p_width}p + {h_width}h). "
                f"Inference would be mathematically invalid."
            )

        # 3. Logic: Vocabulary (Embedding Mismatch)
        if obs_tokens > self.expected_vocab:
            raise ValueError(
                f"🔢 VOCAB MISMATCH: HDF5 contains token ID {obs_tokens-1}, "
                f"but model only supports up to {self.expected_vocab-1}."
            )

        print(f"🧪 Data Validation Passed: {Path(features_path).name}")

    @torch.no_grad()
    def predict(self, features_path: str, labels_path: str):
        # Always validate before starting the loop
        self.validate_hdf5(features_path)

        dataset = BinaryPairkeyFeatureDataset(
            cfg=self.model.cfg, features_path=features_path, labels_path=labels_path
        )

        loader = DataLoader(
            dataset, batch_size=self.model.cfg.batch_size, shuffle=False, num_workers=0
        )

        all_probs = []
        for batch in loader:
            x, _ = batch
            logits = self.model(x.to(self.device))
            all_probs.append(torch.softmax(logits, dim=-1).cpu().numpy())

        if not all_probs:
            return np.array([])

        return np.concatenate(all_probs, axis=0)
