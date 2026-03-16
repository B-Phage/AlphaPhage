from pathlib import Path

import h5py
import numpy as np
from data_contracts.h5_schema import H5DataKeys


class H5TrainingWriter:
    def __init__(self, output_dir: Path, base_filename: str):
        self.keys = H5DataKeys()
        self.x_path = output_dir / f"x_{base_filename}.h5"
        self.cl_path = output_dir / f"cl_{base_filename}.h5"

    def write(self, data_dict: dict, phage_info: np.ndarray, host_info: np.ndarray):
        """
        Writes relational H5 files.
        data_dict should now include:
        - 'phage_vecs': Unique features (M rows)
        - 'host_vecs': Unique features (K rows)
        - 'pair_indices': Mapped integers (N rows x 2)
        - 'labels': Interaction results (N rows)
        """

        # 1. Write Feature File (X)
        with h5py.File(self.x_path, "w") as f:
            # These are your unique "Libraries"
            f.create_dataset(self.keys.PHAGE_VEC, data=data_dict["phage_vecs"])
            f.create_dataset(self.keys.HOST_VEC, data=data_dict["host_vecs"])

            # This is your "Relational Map" (Length N)
            f.create_dataset(self.keys.PAIR_INDICES, data=data_dict["pair_indices"])

            # Metadata
            f.create_dataset(self.keys.PAIR_KEYS, data=data_dict["pair_keys"])
            f.create_dataset(self.keys.PHAGE_VEC_INFO, data=phage_info)
            f.create_dataset(self.keys.HOST_VEC_INFO, data=host_info)

        # 2. Write Label File (CL)
        with h5py.File(self.cl_path, "w") as f:
            # Labels match the length of pair_indices (Length N)
            f.create_dataset("data", data=data_dict["labels"])

        return self.x_path, self.cl_path
