from typing import Dict, Optional

import numpy as np
import pandas as pd
from variants.cluster_map import rigorous_assignment
from variants.distance import calculate_distance


class InferenceMapper:
    def __init__(self, buffer_factor: float = 1.1, consensus_ratio: float = 1.2):
        """
        Args:
            buffer_factor: Multiplier for the stored radius to define the
                'Unknown' boundary (e.g., 1.1 = 10% buffer).
            consensus_ratio: Threshold to allow k-NN consensus to override 1-NN.
                1.2 means consensus must be within 20% of the best distance.
        """
        self.buffer = buffer_factor
        self.consensus_ratio = consensus_ratio

    def map_to_variants(
        self, queries_df: pd.DataFrame, gallery_df: pd.DataFrame, matrix_type: str
    ) -> pd.DataFrame:
        """
        Maps new sequences to existing variant_ids using distance-weighted consensus.
        """
        # 1. Exact Match Shortcut
        shortcut_map = dict(zip(gallery_df["sequence_aligned"], gallery_df["variant_id"]))

        results = []
        to_calculate_indices = []

        for idx, row in queries_df.iterrows():
            seq = row["sequence_aligned"]
            if seq in shortcut_map:
                results.append(
                    {
                        "organism_id": row["organism_id"],
                        "variant_id": shortcut_map[seq],  # Renamed
                        "sequence_aligned": seq,  # Added
                        "distance": 0.0,
                        "mapping_method": "exact_match",
                    }
                )
            else:
                to_calculate_indices.append(idx)

        # 2. Distance Matrix Calculation
        if to_calculate_indices:
            miss_df = queries_df.loc[to_calculate_indices]

            # Use byte-arrays for optimized engine performance
            q_matrix = np.array([list(s) for s in miss_df["sequence_aligned"]], dtype="|S1")
            g_matrix = np.array([list(s) for s in gallery_df["sequence_aligned"]], dtype="|S1")

            distances = calculate_distance(query=q_matrix, target=g_matrix, matrix=matrix_type)

            # 3. Apply Rigorous Weighted Assignment
            gallery_ids = gallery_df["variant_id"].values

            # Map radii to variant_ids for the assignment logic
            # We pre-apply the buffer_factor here
            thresholds = {
                vid: rad * self.buffer
                for vid, rad in zip(gallery_df["variant_id"], gallery_df["radius"])
            }

            assigned_ids, confidence_scores = rigorous_assignment(
                distances=distances,
                gallery_ids=gallery_ids,
                cluster_thresholds=thresholds,
                consensus_ratio=self.consensus_ratio,
            )

            for i, q_idx in enumerate(to_calculate_indices):
                results.append(
                    {
                        "organism_id": queries_df.loc[q_idx, "organism_id"],
                        "variant_id": assigned_ids[i],
                        "sequence_aligned": queries_df.loc[q_idx, "sequence_aligned"],
                        "distance": confidence_scores[i],
                        "mapping_method": f"distance_{matrix_type}",
                    }
                )

        return pd.DataFrame(results)
