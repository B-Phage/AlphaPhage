import numpy as np


def rigorous_assignment(distances, gallery_ids, cluster_thresholds=None, consensus_ratio=1.2):
    """
    Args:
        distances: M x N matrix (M queries, N gallery sequences)
        gallery_ids: Variant IDs for the N gallery sequences
        cluster_thresholds: Dict {variant_id: max_allowed_dist}
        consensus_ratio: Only allow consensus to override 1-NN if the
            consensus distance is within X% of the best distance.
    """
    assigned_variants = []
    confidence_scores = []

    K = min(3, distances.shape[1])
    top_k_indices = np.argpartition(distances, K - 1, axis=1)[:, :K]

    for i, indices in enumerate(top_k_indices):
        row_distances = distances[i, indices]
        sorted_rel_idx = np.argsort(row_distances)

        # 1-NN Info
        best_match_idx = indices[sorted_rel_idx[0]]
        best_dist = distances[i, best_match_idx]
        best_variant = gallery_ids[best_match_idx]

        # --- LOGIC 1: Absolute Threshold Check ---
        is_within_limit = True
        if cluster_thresholds and best_variant in cluster_thresholds:
            if best_dist > cluster_thresholds[best_variant]:
                is_within_limit = False
        elif best_dist > 0.5:
            is_within_limit = False

        # --- LOGIC 2: k-NN Weighted Consensus ---
        top_variants = gallery_ids[indices[sorted_rel_idx]]
        unique_variants, counts = np.unique(top_variants, return_counts=True)
        most_freq_idx = np.argmax(counts)
        consensus_variant = unique_variants[most_freq_idx]

        # Find the best distance associated with the consensus variant
        # (e.g., if V2 appears twice, what is its closest distance?)
        consensus_indices = [
            indices[sorted_rel_idx[j]] for j, v in enumerate(top_variants) if v == consensus_variant
        ]
        best_consensus_dist = np.min(distances[i, consensus_indices])

        # --- FINAL DECISION ---
        if not is_within_limit:
            final_id = "UNKNOWN_NEW_VARIANT"
        elif counts[most_freq_idx] >= 2:
            # Check if the consensus is "comparably close" to the 1-NN
            # If 1-NN is much better (ratio > 1.2), ignore the consensus
            if best_consensus_dist <= (best_dist * consensus_ratio):
                final_id = consensus_variant
            else:
                final_id = best_variant  # 1-NN is significantly higher quality
        else:
            final_id = best_variant

        assigned_variants.append(final_id)
        confidence_scores.append(best_dist)

    return assigned_variants, confidence_scores
