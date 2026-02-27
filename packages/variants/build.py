import argparse
import os
from typing import Optional

from omegaconf import DictConfig
from representation.align_table import process_msa_table
from utils.config_loader import get_vocab
from utils.consol_log import get_logger
from variants.clustering import build_cluster_feature_table
from variants.preprocessing_methods import split_names_diversity, standardize_frame_chars

logger = get_logger()


def build_variants(
    in_file: str,
    msa_cluster_dir: str,
    vocab: DictConfig,
    reduction_method: Optional[str] = None,
    num_clusters: int = 16,
    matrix: str = "identity",
):
    """Processes a Multiple Sequence Alignment (MSA) to generate cluster features.

    This function reads an MSA file, standardizes the characters based on a
    provided vocabulary, applies an optional dimensionality reduction to the
    alignment columns, and performs hierarchical clustering. The results are
    saved as two CSV files: a variant table (.mva) and a metadata/info
    table (.mcd).

    Args:
        in_file: Path to the input MSA file (e.g., .fasta or .aln).
        msa_cluster_dir: Directory where the output .mva and .mcd files
            will be saved.
        vocab: A configuration object containing 'filler_char' and
            'vocab_clusters_single' encoding maps.
        reduction_method: The method used to subset MSA columns.
            Supported value is currently 'diversity'. If None, all
            columns are used.
        num_clusters: The fixed number of clusters to generate.
            Defaults to 16.
        matrix: The distance metric used for clustering. Common options
            include 'identity' or 'blosum62'. Defaults to 'identity'.

    Returns:
        None: The function writes outputs directly to disk and logs progress.

    Raises:
        OSError: If the input file cannot be read or the output directory
            cannot be created.
        ValueError: If the MSA contains fewer sequences than `num_clusters`.
    """
    os.makedirs(msa_cluster_dir, exist_ok=True)

    # File path logic
    hmm_name = os.path.basename(in_file)
    hmm, _ = os.path.splitext(hmm_name)
    out_mva = os.path.join(msa_cluster_dir, f"{hmm}.mva")
    out_mcd = os.path.join(msa_cluster_dir, f"{hmm}.mcd")

    logger.info(f"Processing {in_file} | Matrix: {matrix} | Reduction: {reduction_method}")

    # Read and standardize MSA
    df = process_msa_table(in_file, vocab=vocab)
    df, _ = standardize_frame_chars(df, filler_char=vocab.filler_char)
    genomes = list(df.index)

    # Column selection (formerly reduction_methods list logic)
    if reduction_method == "diversity":
        # Assuming split_names_diversity returns a list of lists,
        # we take the first set of columns for a single execution.
        lst_columns = split_names_diversity(df)
        columns_subset = lst_columns[0] if lst_columns else list(df.columns)
    else:
        columns_subset = list(df.columns)

    # Run clustering
    df_clusters, df_cluster_info = build_cluster_feature_table(
        df=df,
        genomes=genomes,
        columns_subset=columns_subset,
        num_clusters=num_clusters,
        matrix=matrix,
        vocab_dict=vocab.vocab_clusters_single,
    )

    # Save outputs
    df_clusters.to_csv(out_mva, sep=",", index=True, header=True)
    df_cluster_info.to_csv(out_mcd, sep=",", index=True, header=True)
    logger.info(f"Clustering complete. Results saved to {msa_cluster_dir}")


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
