from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class H5DataKeys:
    """Standardized keys for the BinaryPairkey HDF5 format."""

    HOST_VEC: str = "host_vec_1"
    PHAGE_VEC: str = "phage_vec_1"
    PAIR_KEYS: str = "pair_keys"
    PAIR_INDICES: str = "pair_indices"
    LABELS: str = "data"
    HOST_VEC_INFO: str = "host_vec_1_info"
    PHAGE_VEC_INFO: str = "phage_vec_1_info"

    PAIR_INT: str = "data"


@dataclass(frozen=True)
class TrainingManifest:
    """Schema for the metadata describing a generated H5 dataset."""

    phage_dim: int
    host_dim: int
    num_samples: int
    phage_table: str
    host_table: str
    serotype_table: str
    vocab_hash: str
