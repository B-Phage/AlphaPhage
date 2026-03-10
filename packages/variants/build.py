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
    df_clusters, _ = build_cluster_feature_table(
        df=df,
        genomes=genomes,
        columns_subset=columns_subset,
        num_clusters=num_clusters,
        matrix=matrix,
        vocab_dict=vocab.vocab_clusters_single,
    )
    variant_assignments = df_clusters.index.tolist()

    red_method = reduction_method if reduction_method else "none"
    clustering_version = f"red-{red_method}_clusters-{num_clusters}_mat-{matrix}"

    shard_df = pd.DataFrame(
        {
            "organism_id": genomes,
            "variant_id": variant_assignments,
            "hmm_id": hmm_id,
            "clustering_version": clustering_version,
        }
    )

    # 5. Save to Lake
    write_variant_shard(shard_df, pathlib.Path(out_parquet))
    # Useful for debugging here because parquets are not human readable
    # Save to a temporary CSV in the current directory
    # shard_df.to_csv("debug_shard.csv", index=False)
    logger.info(f"Shard saved: {out_parquet}")


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
