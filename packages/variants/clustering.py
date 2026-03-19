from typing import Dict, List, Literal, Tuple, cast

import numpy as np
import pandas as pd
from representation.encoding_methods import encode_onecol_vocab_vec
from sklearn.cluster import AgglomerativeClustering
from utils.consol_log import get_logger
from variants.distance import calculate_distance_matrix

logger = get_logger()
LinkageType = Literal["ward", "complete", "average", "single"]

# Constants for Biological Floors (1 - similarity)
IDENTITY_FLOOR = 0.05  # Allow ~5% divergence for single-seq clusters
BLOSUM_FLOOR = 0.10  # Normalized BLOSUM distance floor


def _calculate_radii(
    ids: List[str], dist_mtx: np.ndarray, fallback_radius: float
) -> Dict[str, float]:
    """Calculates the maximum internal pairwise distance (radius) for each cluster.

    For each unique ID in the provided list, this function extracts the corresponding
    sub-matrix from the global distance matrix and finds the maximum value. If a
    cluster contains only one member or the calculated distance is below the
    fallback threshold, the biological floor is applied.

    Args:
        ids: A list of final variant identifiers (tokens) for each sequence,
            ordered identically to the rows/columns of the distance matrix.
        dist_mtx: A symmetric NxN numpy array containing pairwise biological
            distances between all sequences in the MSA.
        fallback_radius: The minimum radius to assign (Biological Floor). Used
            to provide a "confidence bubble" around highly conserved or
            single-sequence clusters.

    Returns:
        A dictionary mapping each unique variant ID string to its calculated
        radius (float).
    """
    cluster_radii = {}
    unique_ids = set(ids)

    for uid in unique_ids:
        # Find indices in the original matrix that share this final ID
        indices = [i for i, val in enumerate(ids) if val == uid]

        if len(indices) > 1:
            # Slices the pairwise distances for this specific cluster
            sub_dist = dist_mtx[np.ix_(indices, indices)]
            # Radius is the max distance found, or the biological floor
            cluster_radii[uid] = max(float(np.max(sub_dist)), fallback_radius)
        else:
            # Single sequence clusters always get the floor
            cluster_radii[uid] = fallback_radius
    return cluster_radii


def perform_msa_clustering(
    df_subset: pd.DataFrame, num_clusters: int, matrix: str
) -> Tuple[List[str], str, int, np.ndarray]:
    """Handles core biological clustering and distance matrix computation.

    This function computes a pairwise distance matrix for the unique sequences
    in the subset and performs Agglomerative Clustering. It caps the number of
    clusters at the number of unique sequences available.

    Args:
        df_subset: A DataFrame containing the standardized MSA sequences (rows)
            across selected alignment positions (columns).
        num_clusters: The requested maximum number of clusters (K) to generate.
        matrix: The distance metric/substitution matrix to use.
            Must be one of ['identity', 'blosum62'].

    Returns:
        A tuple containing:
            - labels (List[str]): Raw string labels assigned by the clustering
                algorithm (e.g., "1", "2").
            - algorithm (str): The name of the clustering algorithm used
                (e.g., "agglomerative").
            - final_k (int): The actual number of clusters produced.
            - dist_mtx (np.ndarray): The full NxN pairwise distance matrix
                required for downstream radius calculation.
    """
    num_unique = len(df_subset.drop_duplicates())
    linkage = "average"

    # Compute the matrix once
    dist_mtx = calculate_distance_matrix(df_subset.values, matrix)

    if num_unique <= 1:
        labels = ["1"] * len(df_subset)
        return labels, "single_row_default", 1, dist_mtx

    final_k = min(num_clusters, num_unique)

    # Clustering logic
    model = AgglomerativeClustering(n_clusters=final_k, linkage=linkage, metric="precomputed")
    clusters = model.fit_predict(dist_mtx)
    labels = [str(int(c) + 1) for c in clusters]

    return labels, "agglomerative", final_k, dist_mtx


def build_cluster_feature_table(
    df: pd.DataFrame,
    genomes: List[str],
    columns_subset: List[str],
    num_clusters: int,
    matrix: str,
    vocab_dict: Dict[str, int],
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    """Orchestrates MSA clustering, vocabulary encoding, and radius calculation.

    This is the primary entry point for transforming an alignment into a
    tokenized feature table. It ensures that the calculated radii are
    correctly mapped to the final vocabulary tokens, preventing key mismatches
    between the variant gallery and the distance thresholds.

    Args:
        df: The input MSA DataFrame where rows represent genomes and columns
            represent alignment positions.
        genomes: A list of unique identifiers (e.g., organism_id) corresponding
            to the rows in `df`.
        columns_subset: The list of alignment positions (columns) used as
            features for clustering.
        num_clusters: Target number of clusters to generate.
        matrix: The distance metric to use for pairwise comparisons
            ('identity' or 'blosum62').
        vocab_dict: A dictionary mapping raw cluster labels to integer
            tokens (e.g., {'1': 3, '2': 4}).

    Returns:
        A tuple containing:
            - df_clusters (pd.DataFrame): The original sequence data with an
                index consisting of the tokenized cluster IDs.
            - df_cluster_info (pd.DataFrame): Summary table of the clustering
                parameters (algorithm, linkage, etc.) for metadata tracking.
            - radii (Dict[str, float]): A mapping of final variant tokens
                to their associated radii (max internal distance or floor).
    """

    # 1. Cluster and get the Matrix
    cluster_labels, algorithm, final_k, dist_mtx = perform_msa_clustering(
        df[columns_subset], num_clusters, matrix
    )

    # 2. Tokenize (IDs change from "1" to "3", etc.)
    # Note: encode_onecol_vocab_vec is imported from your representation module
    from representation.encoding_methods import encode_onecol_vocab_vec

    nested_tokens = encode_onecol_vocab_vec(vec=cluster_labels, vocab_dict=vocab_dict)
    flat_variant_ids = [str(t) for t in nested_tokens[0]]

    # 3. Calculate Radii using FINAL IDs
    fallback_radius = IDENTITY_FLOOR if matrix == "identity" else BLOSUM_FLOOR
    radii = _calculate_radii(flat_variant_ids, dist_mtx, fallback_radius)

    # 4. Update DataFrame
    df_clusters = df.copy()
    df_clusters.index = flat_variant_ids

    # 5. Metadata
    df_cluster_info = pd.DataFrame(
        [["cluster", algorithm, "average", matrix, final_k, len(columns_subset)]],
        columns=[
            "feature_type",
            "clustering_algorithm",
            "linkage",
            "distance_metric",
            "num_clusters",
            "msa_width",
        ],
    )

    return df_clusters, df_cluster_info, radii
