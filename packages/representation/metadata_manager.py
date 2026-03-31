import numpy as np
import pandas as pd


class MetadataManager:
    @staticmethod
    def build_info_array(
        cols: list, meta_df: pd.DataFrame, source_map: dict | None = None
    ) -> np.ndarray:
        """Constructs the 2-row S100 metadata array."""
        ver_map = dict(zip(meta_df["hmm_id"], meta_df["clustering_version"]))
        row1 = cols
        row2 = (
            [source_map.get(c, str(ver_map.get(c, "unknown"))) for c in cols]
            if source_map
            else [str(ver_map.get(c, "unknown")) for c in cols]
        )

        return np.array([row1, row2]).astype("S100")
