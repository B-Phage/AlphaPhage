import numpy as np
import pandas as pd


class MetadataManager:
    @staticmethod
    def build_info_array(
        cols: list, meta_df: pd.DataFrame, source_map: dict | None = None
    ) -> np.ndarray:
        """
        Constructs a 2-row S100 metadata array.
        Row 0: Unique HMM Labels (the column names)
        Row 1: Method/Clustering Version info
        """
        # Create a lookup for the version info
        ver_map = dict(zip(meta_df["hmm_id"], meta_df["clustering_version"]))

        # Row 1: The labels (HMM IDs)
        row1 = cols

        # Row 2: The metadata (Versions)
        # We handle serotype columns (k_type_dim1/2) as 'serotype_logic'
        # and HMMs as their clustering version.
        row2 = []
        for c in cols:
            if source_map and c in source_map:
                row2.append(source_map[c])
            elif c in ["k_type_dim1", "k_type_dim2"]:
                row2.append("serotype_v1")
            else:
                # Fallback to the meta_df lookup or 'unknown'
                row2.append(str(ver_map.get(c, "unknown")))

        # CRITICAL: Ensure we return a (2, N) array, not a (1, 2N) array
        return np.array([row1, row2]).astype("S100")
