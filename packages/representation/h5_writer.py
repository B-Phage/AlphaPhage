from pathlib import Path

import h5py
import numpy as np
from data_contracts.h5_schema import H5DataKeys


class H5TrainingWriter:
    def __init__(self, output_dir: Path, base_filename: str):
        self.keys = H5DataKeys()
        self.x_path = output_dir / f"x_{base_filename}.h5"
        self.cl_path = output_dir / f"cl_{base_filename}.h5"

    def write(
        self,
        phage_vecs: np.ndarray,
        host_vecs: np.ndarray,
        pair_indices: np.ndarray,
        pair_keys: np.ndarray,
        labels: np.ndarray,
        phage_info: np.ndarray,
        host_info: np.ndarray,
    ):
        """Writes relational H5 files using the internal H5DataKeys contract."""

        # 1. Write Feature File (X)
        with h5py.File(self.x_path, "w") as f:
            f.create_dataset(self.keys.PHAGE_VEC, data=phage_vecs)
            f.create_dataset(self.keys.HOST_VEC, data=host_vecs)
            f.create_dataset(self.keys.PAIR_INDICES, data=pair_indices)
            f.create_dataset(self.keys.PAIR_KEYS, data=pair_keys)
            f.create_dataset(self.keys.PHAGE_VEC_INFO, data=phage_info)
            f.create_dataset(self.keys.HOST_VEC_INFO, data=host_info)

        # 2. Write Label File (CL)
        with h5py.File(self.cl_path, "w") as f:
            # Note: Using keys.LABELS from your contract instead of hardcoded "data"
            f.create_dataset(self.keys.LABELS, data=labels)

        return self.x_path, self.cl_path
