import argparse
import os
import pathlib
from typing import Optional

import pandas as pd
from omegaconf import DictConfig
from representation.align_table import process_msa_table
from utils.config_loader import get_vocab
from utils.consol_log import get_logger
from variants.clustering import build_cluster_feature_table
from variants.preprocessing_methods import split_names_diversity, standardize_frame_chars

logger = get_logger()


from variants.variant_lake_writer import write_variant_shard


def build_variants(
    in_file: str,
    msa_cluster_dir: str,
    vocab: DictConfig,
    reduction_method: Optional[str] = None,
    num_clusters: int = 16,
    matrix: str = "identity",
):
    """Processes an MSA file to generate clusters and writes raw variant shards.

    This function executes the full training preprocessing pipeline for a single HMM.
    It reads an MSA, standardizes characters, optionally reduces the feature space
    via diversity partitioning, and performs agglomerative clustering.

    The resulting assignments, along with the aligned sequences and calculated
    cluster radii, are persisted as a Parquet shard. These shards serve as the
    reference 'gallery' for future inference mapping.

    Args:
        in_file: Path to the input Multiple Sequence Alignment (MSA) file
            (typically named after the HMM ID).
        msa_cluster_dir: Directory where the resulting Parquet shards should
            be saved.
        vocab: Configuration object containing vocabulary mappings and
            special characters (e.g., filler_char).
        reduction_method: Optional method to subset MSA columns. Currently
            supports 'diversity'. If None, the full alignment is used.
        num_clusters: Target number of clusters to generate for the HMM.
            Defaults to 16.
        matrix: The distance metric used for clustering ('identity' or 'blosum62').
            Defaults to 'identity'.

    Returns:
        None. The results are written directly to a Parquet file in
        `msa_cluster_dir`.

    Notes:
        The output Parquet shard contains the following schema:
        - organism_id (str): Unique identifier for the source genome.
        - hmm_id (str): The HMM identifier.
        - variant_id (str): The RAW cluster label (e.g., "0", "1").
            Note: These are mapped to global vocabulary tokens in a
            separate downstream process.
        - sequence_aligned (str): The raw aligned sequence.
        - radius (float): Max internal distance of the cluster (biological floor applied).
        - clustering_version (str): Metadata for reproducibility.
    """
    msa_cluster_dir = str(msa_cluster_dir)
    os.makedirs(msa_cluster_dir, exist_ok=True)

    # 1. Extract Metadata from filename
    hmm_id = pathlib.Path(in_file).stem
    out_parquet = os.path.join(msa_cluster_dir, f"{hmm_id}.parquet")

    logger.info(f"Processing {hmm_id} | Matrix: {matrix}")

    # 2. Existing processing logic (unchanged)
    df = process_msa_table(in_file, vocab=vocab)
    df, _ = standardize_frame_chars(df, filler_char=vocab.filler_char)
    genomes = list(df.index)

    if reduction_method == "diversity":
        lst_columns = split_names_diversity(df)
        columns_subset = lst_columns[0] if lst_columns else list(df.columns)
    else:
        columns_subset = list(df.columns)

    # 3. Generate Clustering
    # df_clusters: index is organism, columns are the variant assignments
    df_clusters, _, cluster_radii = build_cluster_feature_table(
        df=df,
        genomes=genomes,
        columns_subset=columns_subset,
        num_clusters=num_clusters,
        matrix=matrix,
    )
    # These are the raw cluster strings from the clustering index
    variant_assignments = df_clusters.index.tolist()

    try:
        # Validates that every row's cluster label has a corresponding radius entry
        row_radii = [cluster_radii[v] for v in variant_assignments]
    except KeyError as e:
        logger.error(f"❌ Radius Lookup Failure: Label {e} not found in radii dict.")
        raise

    # Convert MSA dataframe back to raw strings for the "Gallery"
    aligned_sequences = df.apply(lambda x: "".join(x), axis=1).tolist()

    red_method = reduction_method if reduction_method else "none"
    clustering_version = f"red-{red_method}_clusters-{num_clusters}_mat-{matrix}"

    shard_df = pd.DataFrame(
        {
            "organism_id": genomes,
            "variant_id": variant_assignments,
            "hmm_id": hmm_id,
            "sequence_aligned": aligned_sequences,  # Required for distance mapping
            "radius": row_radii,  # Required for unknown detection
            "clustering_version": clustering_version,
        }
    )

    # 5. Save to Lake
    write_variant_shard(shard_df, pathlib.Path(out_parquet))
    # Useful for debugging here because parquets are not human readable
    # Save to a temporary CSV in the current directory
    # shard_df.to_csv("debug_shard.csv", index=False)
    logger.info(f"Rigorous Shard saved for {hmm_id} with {len(cluster_radii)} clusters.")


def cli():
    """Command-line interface for build_variants."""
    parser = argparse.ArgumentParser(description="Build cluster features from MSA files.")
    parser.add_argument("-f", "--in_file", type=str, required=True, help="Path to input MSA file")
    parser.add_argument("-o", "--msa_cluster_dir", type=str, required=True, help="Output directory")
    parser.add_argument("-c", "--num_clusters", type=int, default=16, help="Number of clusters")

    # Changed from nargs='*' to single string inputs with choices
    parser.add_argument(
        "-r",
        "--reduction_method",
        type=str,
        choices=["diversity"],
        help="Optional reduction method",
    )
    parser.add_argument(
        "-m",
        "--matrix",
        type=str,
        default="identity",
        choices=["identity", "blosum62"],
        help="Distance matrix type",
    )

    args = parser.parse_args()

    vocab = get_vocab()
    build_variants(
        in_file=args.in_file,
        msa_cluster_dir=args.msa_cluster_dir,
        vocab=vocab,
        reduction_method=args.reduction_method,
        num_clusters=args.num_clusters,
        matrix=args.matrix,
    )


if __name__ == "__main__":
    cli()
