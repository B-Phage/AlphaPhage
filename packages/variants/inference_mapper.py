from typing import Dict, List, cast

import numpy as np
import pandas as pd
from variants.cluster_map import rigorous_assignment
from variants.distance import calculate_distance


class InferenceMapper:
    def __init__(
        self, buffer_factor: float = 1.1, consensus_ratio: float = 1.2, fallback_radius: float = 0.5
    ):
        self.buffer = buffer_factor
        self.consensus_ratio = consensus_ratio
        self.fallback_radius = fallback_radius

    def map_to_variants(
        self, queries_df: pd.DataFrame, gallery_df: pd.DataFrame, matrix_type: str, hmm_id: str
    ) -> pd.DataFrame:
        """
        Consumes the split DataFrames from the PreProcessor.

        Args:
            queries_df: Aligned inference sequences.
            gallery_df: Aligned training sequences (with variant_id and radius).
            matrix_type: 'identity' or 'blosum62'.
            hmm_id: The HMM identifier for the final contract.
        """

        # 1. Exact Match Shortcut
        # We map sequence -> variant_id. Since the PreProcessor merged variant_id
        # back into the gallery_df, this is a clean lookup.
        shortcut_map = dict(
            zip(gallery_df["sequence_aligned"], gallery_df["variant_id"].astype(str))
        )

        results = []
        to_calculate_indices = []

        # itertuples is faster for large inference batches in the orchestrator
        for row in queries_df.itertuples():
            seq = row.sequence_aligned
            if seq in shortcut_map:
                results.append(
                    {
                        "organism_id": row.organism_id,
                        "variant_id": shortcut_map[seq],
                        "sequence_aligned": seq,
                        "distance": 0.0,
                        "mapping_method": "exact_match",
                        "hmm_id": hmm_id,
                    }
                )
            else:
                # Store the index to pull from queries_df later
                to_calculate_indices.append(row.Index)

        # 2. Distance Matrix Calculation for Misses
        if to_calculate_indices:
            miss_df = queries_df.loc[to_calculate_indices]

            # Vectorized conversion to byte arrays
            q_matrix = np.array([list(s) for s in miss_df["sequence_aligned"]], dtype="|S1")
            g_matrix = np.array([list(s) for s in gallery_df["sequence_aligned"]], dtype="|S1")

            distances = calculate_distance(query=q_matrix, target=g_matrix, matrix=matrix_type)

            # 3. Apply Rigorous Weighted Assignment
            gallery_ids = gallery_df["variant_id"].astype(str).values

            # Apply buffer to stored radii
            thresholds = {
                str(vid): float(rad) * self.buffer
                for vid, rad in zip(gallery_df["variant_id"], gallery_df["radius"])
            }

            assigned_ids, confidence_scores = rigorous_assignment(
                distances=distances,
                gallery_ids=gallery_ids,
                cluster_thresholds=thresholds,
                consensus_ratio=self.consensus_ratio,
                fallback_radius=self.fallback_radius,
            )

            # Zip results back to the original misses
            for i, (idx, row) in enumerate(miss_df.iterrows()):
                results.append(
                    {
                        "organism_id": row["organism_id"],
                        "variant_id": str(assigned_ids[i]),
                        "sequence_aligned": row["sequence_aligned"],
                        "distance": float(confidence_scores[i]),
                        "mapping_method": f"distance_{matrix_type}",
                        "hmm_id": hmm_id,
                    }
                )

        return pd.DataFrame(results)
