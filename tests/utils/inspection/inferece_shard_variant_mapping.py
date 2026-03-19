#!/usr/bin/env python3
import argparse
import pathlib
import sys
from typing import Tuple

import numpy as np
import pandas as pd
from utils.consol_log import get_logger
from variants.inference_mapper import InferenceMapper

logger = get_logger()


def prepare_single_shard(
    shard_path: str, num_queries: int = 5
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads one shard and splits it into a Gallery and a Query set.
    """
    try:
        # Handling both TSV and Parquet since build.py output is Parquet
        if shard_path.endswith(".parquet"):
            df = pd.read_parquet(shard_path)
        else:
            df = pd.read_table(shard_path)
    except Exception as e:
        logger.error(f"Failed to load shard at {shard_path}: {e}")
        sys.exit(1)

    if len(df) <= num_queries:
        logger.warning(f"Shard only has {len(df)} rows. Reducing queries to 1.")
        num_queries = 1

    # Shuffle to ensure queries are randomly selected
    df = df.sample(frac=1.0).reset_index(drop=True)

    queries = df.head(num_queries).copy()
    gallery = df.tail(len(df) - num_queries).copy()

    logger.info(f"--- Shard Prepared: {pathlib.Path(shard_path).name} ---")
    logger.info(f"Gallery Size: {len(gallery)} | Test Queries: {len(queries)}")
    return queries, gallery


def run_sanity_check(shard_path: str, num_q: int, matrix: str, buffer: float, ratio: float):
    # 1. Prepare Data
    queries, gallery = prepare_single_shard(shard_path, num_queries=num_q)

    # 2. Initialize Mapper with your specific Rigorous Logic
    mapper = InferenceMapper(buffer_factor=buffer, consensus_ratio=ratio)

    # 3. Perform Mapping
    results = mapper.map_to_variants(queries, gallery, matrix_type=matrix)

    # 4. Detailed Output
    print("\n" + "=" * 90)
    print(f"{'ORGANISM_ID':<25} | {'ASSIGNED_ID':<15} | {'DIST':<8} | {'METHOD'}")
    print("-" * 90)

    for _, row in results.iterrows():
        is_unknown = row["assigned_variant_id"] == "UNKNOWN_NEW_VARIANT"
        status_tag = "[!]" if is_unknown else "[OK]"

        # Color-coded feel for terminals (optional)
        print(
            f"{row['organism_id'][:25]:<25} | "
            f"{row['assigned_variant_id']:<15} | "
            f"{row['distance']:<8.4f} | "
            f"{status_tag} {row['mapping_method']}"
        )

    print("=" * 90)

    # 5. Summary
    total = len(results)
    unknowns = (results["assigned_variant_id"] == "UNKNOWN_NEW_VARIANT").sum()
    logger.info(
        f"Results for {pathlib.Path(shard_path).stem}: {total - unknowns}/{total} mapped successfully."
    )


if __name__ == "__main__":
    #

    # parser = argparse.ArgumentParser(description="Sanity check for Variant Mapping on a single shard.")
    # parser.add_argument("-f", "--file", type=str, required=True, help="Path to shard (TSV or Parquet)")
    # parser.add_argument("-n", "--num_queries", type=int, default=5, help="Number of rows to treat as 'New'")
    # parser.add_argument("-m", "--matrix", type=str, default="identity", choices=["identity", "blosum62"])
    # parser.add_argument("-b", "--buffer", type=float, default=1.1, help="Radius buffer (default 1.1)")
    # parser.add_argument("-r", "--ratio", type=float, default=1.2, help="Consensus ratio (default 1.2)")

    # args = parser.parse_args()

    # run_sanity_check(
    #     shard_path=args.file,
    #     num_q=args.num_queries,
    #     matrix=args.matrix,
    #     buffer=args.buffer,
    #     ratio=args.ratio
    # )

    # file_name = "shard_Wzy_C.tsv"
    # num_queries = 300
    file_name = "shard_phrog_17.tsv"
    num_queries = 50
    matrix = "blosum62"
    radius_buffer = 1.1
    consensus_ratio = 1.1
    run_sanity_check(
        shard_path=file_name,
        num_q=num_queries,
        matrix=matrix,
        buffer=radius_buffer,
        ratio=consensus_ratio,
    )
