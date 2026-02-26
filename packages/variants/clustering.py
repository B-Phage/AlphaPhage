from typing import Dict, Literal, cast

import numpy as np
import pandas as pd
from features.encoding_methods import encode_onecol_vocab_vec
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
    vocab_dict: Dict[str, int],
):
    """
    Build a tokenised dataframe representing sequence clusters for a single
    clustering configuration (>= 2 clusters required).

    Parameters
    ----------
    df : pd.DataFrame
        MSA dataframe (rows = genomes, columns = alignment positions).
    filename : str
        Source filename used for metadata tracking.
    genomes : list[str]
        Genome identifiers corresponding to df rows.
    columns_subset : list[str]
        Subset of MSA columns used for clustering.
    num_clusters : int
        Number of clusters to generate (must be >= 2).
    matrix : str
        Distance matrix to use. Must be one of ['identity', 'blosum62'].
    vocab_dict : Dict[str, int]
        Vocabulary used for token encoding.

    Returns
    -------
    df_clusters : pd.DataFrame
        Dataframe indexed by cluster token combinations, containing
        original sequence rows.
    df_cluster_info : pd.DataFrame
        Metadata describing the clustering configuration.
    """

    if matrix not in ["identity", "blosum62"]:
        raise ValueError("matrix must be 'identity' or 'blosum62'")

    if not isinstance(num_clusters, int):
        raise TypeError("num_clusters must be an integer")

    # Short circuit
    df_subset = df[columns_subset]
    num_unique_samples = len(df_subset.drop_duplicates())
    if num_unique_samples <= 1:
        logger.info(f"Only {num_unique_samples} unique alignment(s) found. Skipping clustering.")

        # Manually create the cluster label "1"
        # We encode it using the vocab_dict to match your expected token format
        # If '1' isn't a string key in your vocab, adjust to match your logic
        cluster_label = encode_onecol_vocab_vec(vec=["1"], vocab_dict=vocab_dict)

        # Build the outputs manually to mirror the clustering logic
        if isinstance(cluster_label[0], list):
            cluster_label = cluster_label[0]

        index_val = ";".join(str(e) for e in cluster_label)

        df_clusters = pd.DataFrame(df.values, index=[index_val] * len(df), columns=list(df.columns))

        df_cluster_info = pd.DataFrame(
            [["cluster", "single_row_default", "none", matrix, 1, len(columns_subset)]],
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

    if num_clusters < 2:
        raise ValueError(
            "num_clusters must be >= 2. " "Presence/absence encoding should use a separate method."
        )

    if not df.index.is_unique:
        duplicates = df.index[df.index.duplicated()].unique().tolist()
        raise ValueError(
            f"Input MSA contains duplicate genome identifiers: {duplicates}. "
            f"Cluster generation requires unique sequence IDs."
        )

    linkage = "average"
    gen_method = "agglomerate"

    columns_all = list(df.columns)
    df_subset = df[columns_subset]

    num_unique_samples = len(df_subset.value_counts())

    logger.info(f"Unique samples = {num_unique_samples}")

    # Prevent requesting more clusters than unique samples
    if num_clusters > num_unique_samples:
        logger.info(
            f"Requested {num_clusters} clusters but only "
            f"{num_unique_samples} unique samples available. "
            f"Reducing to {num_unique_samples}."
        )
        num_clusters = num_unique_samples

    logger.info(f"Number of clusters = {num_clusters}")

    clusters = cluster_aggregate(
        df_subset.values,
        genomes,
        matrix,
        linkage,
        num_clusters,
    )

    cluster_strs = [str(int(c) + 1) for c in clusters]
    vec_out = encode_onecol_vocab_vec(vec=cluster_strs, vocab_dict=vocab_dict)

    # Simplified metadata
    info_out = [
        [
            "cluster",  # feature_type
            "agglomerative",  # clustering_algorithm
            linkage,  # linkage
            matrix,  # distance_metric
            num_clusters,  # number of clusters
            len(columns_subset),  # msa_width
        ]
    ]

    # Build cluster dataframe
    df_clusters = pd.DataFrame(np.array(vec_out).transpose())
    df_clusters.index = genomes

    lst_cluster_aggregate = []
    lst_cluster_vals = []

    for genome in genomes:
        row = df_clusters.loc[genome]
        cluster_vals = ";".join(str(e) for e in row)
        lst_cluster_aggregate.append(df.loc[genome])
        lst_cluster_vals.append(cluster_vals)

    df_clusters = pd.DataFrame(
        lst_cluster_aggregate,
        index=lst_cluster_vals,
        columns=columns_all,
    )

    df_cluster_info = pd.DataFrame(
        info_out,
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
