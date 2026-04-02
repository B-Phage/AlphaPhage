import numpy as np


def rigorous_assignment(
    distances, gallery_ids, cluster_thresholds=None, consensus_ratio=1.2, fallback_radius=0.5
):
    """
    Maps query sequences to variant IDs using a combination of distance thresholds
    and k-Nearest Neighbor (k-NN) consensus.

    Args:
        distances: M x N matrix (M queries, N gallery sequences) of pairwise distances.
        gallery_ids: Array of variant ID strings for the N gallery sequences.
        cluster_thresholds: Dict {variant_id: max_allowed_dist} defining the
            radius for each known cluster.
        consensus_ratio: Sensitivity for overriding 1-NN. If the consensus
            distance is within this ratio of the best distance, consensus wins.

    Returns:
        tuple: (assigned_variants, confidence_scores)
            - assigned_variants: List of strings (e.g., '16', 'UNK').
            - confidence_scores: List of floats representing the 1-NN distance.
    """
    assigned_variants = []
    confidence_scores = []

    # Ensure we don't look for more neighbors than exist in the gallery
    K = min(3, distances.shape[1])
    # Partition to find the indices of the K smallest distances
    top_k_indices = np.argpartition(distances, K - 1, axis=1)[:, :K]

    for i, indices in enumerate(top_k_indices):
        row_distances = distances[i, indices]
        sorted_rel_idx = np.argsort(row_distances)

        # 1-NN Info (The absolute closest match)
        best_match_idx = indices[sorted_rel_idx[0]]
        best_dist = distances[i, best_match_idx]
        best_variant = gallery_ids[best_match_idx]

        # --- LOGIC 1: Absolute Threshold Check ---
        # Determines if the sequence is "known" based on cluster radii
        is_within_limit = True
        if cluster_thresholds and best_variant in cluster_thresholds:
            if best_dist > cluster_thresholds[best_variant]:
                is_within_limit = False
        elif best_dist > fallback_radius:
            # Fallback for clusters missing a specific radius
            is_within_limit = False

        # --- LOGIC 2: k-NN Weighted Consensus ---
        # Get the variants of the top K matches
        top_variants = gallery_ids[indices[sorted_rel_idx]]
        unique_variants, counts = np.unique(top_variants, return_counts=True)
        most_freq_idx = np.argmax(counts)
        consensus_variant = unique_variants[most_freq_idx]

        # Calculate the best distance for the consensus choice
        consensus_indices = [
            indices[sorted_rel_idx[j]] for j, v in enumerate(top_variants) if v == consensus_variant
        ]
        best_consensus_dist = np.min(distances[i, consensus_indices])

        # --- FINAL DECISION ---
        if not is_within_limit:
            # FIX: Return 'UNK' string key to match vocabulary keys,
            # NOT the token integer '2' or the cluster string '2'.
            final_id = "UNK"
        elif counts[most_freq_idx] >= 2:
            # If there's a consensus, check if it's "comparably close" to 1-NN
            if best_consensus_dist <= (best_dist * consensus_ratio):
                final_id = consensus_variant
            else:
                # 1-NN is significantly closer/higher quality than the consensus
                final_id = best_variant
        else:
            # No consensus found among top K; default to 1-NN
            final_id = best_variant

        assigned_variants.append(final_id)
        confidence_scores.append(best_dist)

    return assigned_variants, confidence_scores
