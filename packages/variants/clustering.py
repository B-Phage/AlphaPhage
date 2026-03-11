from typing import Dict, Literal, cast

import numpy as np
import pandas as pd
from representation.encoding_methods import encode_onecol_vocab_vec
from sklearn.cluster import AgglomerativeClustering  # For hierarchical clustering
from utils.consol_log import get_logger
from variants.distance import calculate_distance_matrix

logger = get_logger()
LinkageType = Literal["ward", "complete", "average", "single"]


def build_cluster_feature_table(
    df: pd.DataFrame,
    genomes: list[str],
    columns_subset: list[str],
    num_clusters: int,
    matrix: str,
    vocab_dict: dict[str, int],
):
    """Builds a tokenized dataframe representing sequence clusters.

    This function performs agglomerative clustering on a subset of MSA columns.
    It handles edge cases where the number of unique sequences is less than the
    requested cluster count and encodes the resulting cluster assignments using
    a provided vocabulary.

    Args:
        df: The input Multiple Sequence Alignment (MSA) dataframe where rows
            represent genomes and columns represent alignment positions.
        genomes: A list of unique identifiers corresponding to the rows in `df`.
        columns_subset: The list of column names to be used as features for
            the clustering algorithm.
        num_clusters: The target number of clusters to generate. If this
            exceeds the number of unique sequences, it is capped at the
            unique sequence count.
        matrix: The distance metric/substitution matrix to use.
            Must be one of ['identity', 'blosum62'].
        vocab_dict: A dictionary mapping cluster labels (as strings) to
            integer tokens for encoding.

    Returns:
        A tuple containing:
            - df_clusters (pd.DataFrame): A dataframe containing the original
                sequence data, but indexed by the semicolon-delimited
                tokenized cluster assignments.
            - df_cluster_info (pd.DataFrame): A single-row dataframe containing
                metadata about the clustering run (algorithm, linkage,
                metric, etc.).

    Raises:
        ValueError: If `matrix` is invalid, `num_clusters` is < 2 (and
            multiple unique sequences exist), or if the `df` index contains
            duplicate genome identifiers.
        TypeError: If `num_clusters` is not an integer.
    """
    # 1. Input Validation
    if matrix not in ["identity", "blosum62"]:
        raise ValueError("matrix must be 'identity' or 'blosum62'")
    if not isinstance(num_clusters, int):
        raise TypeError("num_clusters must be an integer")
    if not df.index.is_unique:
        duplicates = df.index[df.index.duplicated()].unique().tolist()
        raise ValueError(f"Duplicate genome identifiers found: {duplicates}")

    # 2. Data Prep & Unique Check
    df_subset = df[columns_subset]
    unique_rows = df_subset.drop_duplicates()
    num_unique = len(unique_rows)
    linkage = "average"

    # 3. Handle Edge Cases (Short-circuit or Cluster adjustment)
    if num_unique <= 1:
        logger.info(f"Only {num_unique} unique alignment(s) found. Skipping.")
        cluster_labels = ["1"] * len(df)
        final_num_clusters = 1
        algorithm = "single_row_default"
    else:
        if num_clusters < 2:
            raise ValueError("num_clusters must be >= 2 for standard clustering.")

        final_num_clusters = min(num_clusters, num_unique)
        if final_num_clusters < num_clusters:
            logger.info(f"Reducing clusters from {num_clusters} to {final_num_clusters}")

        # Execute Clustering
        clusters = cluster_aggregate(df_subset.values, genomes, matrix, linkage, final_num_clusters)
        cluster_labels = [str(int(c) + 1) for c in clusters]
        algorithm = "agglomerative"

    # 4. Vectorized Tokenization & Indexing
    # This returns [[int, int, int]]
    nested_tokens = encode_onecol_vocab_vec(vec=cluster_labels, vocab_dict=vocab_dict)

    # Flatten the outer list to get [int, int, int]
    # Each integer corresponds to a genome in the same order
    flat_tokens = nested_tokens[0]

    # Assign the tokens as strings to the index
    df_clusters = df.copy()
    df_clusters.index = [str(t) for t in flat_tokens]

    # 5. Metadata Construction
    df_cluster_info = pd.DataFrame(
        [["cluster", algorithm, linkage, matrix, final_num_clusters, len(columns_subset)]],
        columns=[
            "feature_type",
            "clustering_algorithm",
            "linkage",
            "distance_metric",
            "num_clusters",
            "msa_width",
        ],
    )

    return df_clusters, df_cluster_info


def cluster_aggregate(
    alignment: np.ndarray,
    index: list[str],
    matrix: str,
    linkage: str = "average",
    num_clusters: int = 2,
) -> pd.Series:
    """
    Perform agglomerative clustering on a multiple sequence alignment (MSA).

    Parameters
    ----------
    alignment : np.ndarray
        2D array of sequences (rows = sequences, columns = positions).
    index : list[str]
        Sequence identifiers corresponding to alignment rows.
    matrix : str
        Distance matrix type. Must be one of ['identity', 'blosum62'].
    linkage : str, default 'average'
        Linkage method for agglomerative clustering ('average', 'ward', 'complete', 'single').
    num_clusters : int, default 2
        Number of clusters to generate (must be >= 2).

    Returns
    -------
    pd.Series
        Cluster assignments for each sequence, indexed by `index`.
    """
    # Validate inputs
    if matrix not in ["identity", "blosum62"]:
        raise ValueError("matrix must be 'identity' or 'blosum62'")
    if not isinstance(num_clusters, int) or num_clusters < 2:
        raise ValueError("num_clusters must be an integer >= 2")
    if not isinstance(alignment, np.ndarray):
        alignment = np.array(alignment)
    if len(index) != alignment.shape[0]:
        raise ValueError("Length of index must match number of alignment rows")

    # Step 1: Compute distance matrix
    distance_matrix = calculate_distance_matrix(alignment, matrix)
    df_distance_matrix = pd.DataFrame(distance_matrix, index=index, columns=index)

    # Step 2: Perform agglomerative clustering
    clusters = cluster_sequences_agglomerate(df_distance_matrix, num_clusters, linkage)

    # Step 3: Return as a pandas Series for easy indexing
    cluster_series = pd.Series(clusters, index=index, name="cluster")
    logger.debug(f"Cluster assignments:\n{cluster_series}")

    return cluster_series


def cluster_sequences_agglomerate(
    distance_matrix: pd.DataFrame | np.ndarray, n_clusters: int = 3, linkage: str = "average"
) -> list[str]:
    """
    Clusters sequences using agglomerative clustering.

    Parameters
    ----------
    distance_matrix : pd.DataFrame or np.ndarray
        Symmetric distance matrix (rows = sequences, columns = sequences)
    n_clusters : int, default 3
        Number of clusters to generate (must be >= 2 and <= number of sequences)
    linkage : str, default 'average'
        Linkage method: 'ward', 'complete', 'average', 'single'.

    Returns
    -------
    List[str]
        Cluster assignments as strings.
    """
    # Validate n_clusters
    n_samples = distance_matrix.shape[0]
    if n_clusters < 2:
        raise ValueError("n_clusters must be >= 2")
    if n_clusters > n_samples:
        logger.warning(
            f"n_clusters={n_clusters} > number of sequences={n_samples}. Reducing to {n_samples}."
        )
        n_clusters = n_samples

    # Cast the generic str to the specific Literal type sklearn expects
    safe_linkage = cast(LinkageType, linkage)

    # 1. Group ONLY identical (or near-identical) sequences
    # 1e-7 absorbs float noise but respects single-AA changes.
    model_identity = AgglomerativeClustering(
        n_clusters=None, distance_threshold=1e-7, linkage=safe_linkage, metric="precomputed"
    )
    clusters = model_identity.fit_predict(distance_matrix)
    num_unique_variants = len(np.unique(clusters))

    # 2. Fit to budget
    if num_unique_variants > n_clusters:
        # Re-run to force the 16 best biological groupings
        model_forced = AgglomerativeClustering(
            n_clusters=n_clusters, linkage=safe_linkage, metric="precomputed"
        )
        clusters = model_forced.fit_predict(distance_matrix)

    return [str(int(c)) for c in clusters]
