#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd
import utils.consol_log as log

logger = log.get_logger()


def build_all_v_all_matrix(phage_dir: Path, host_dir: Path, output_path: Path):
    """
    Constructs a placeholder interaction matrix for inference.

    Format:
        Rows: Host organism IDs (from fasta filenames)
        Cols: Phage organism IDs (from fasta filenames)
        Values: -1 (indicating unknown/untested)
    """
    # 1. Extract IDs from filenames (excluding .fasta)
    phage_ids = sorted([f.stem for f in phage_dir.glob("*.fasta")])
    host_ids = sorted([f.stem for f in host_dir.glob("*.fasta")])

    if not phage_ids or not host_ids:
        raise FileNotFoundError(
            f"Could not find .fasta files. Phages found: {len(phage_ids)}, Hosts found: {len(host_ids)}"
        )

    logger.info(f"🧬 Constructing {len(phage_ids)}x{len(host_ids)} inference matrix...")

    # 2. Create the matrix
    # We initialize with -1. Rows are hosts, Columns are phages.
    df = pd.DataFrame(data=-1, index=host_ids, columns=phage_ids)

    # 3. Save as TSV
    # We include the index so the first column becomes the Host IDs
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, sep="\t", index_label="host_id")

    logger.info(f"✅ Inference interaction matrix saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Build an all-vs-all unknown interaction matrix.")
    parser.add_argument("-p", "--phage_dir", type=Path, required=True)
    parser.add_argument("-b", "--host_dir", type=Path, required=True)
    parser.add_argument("-o", "--output_path", type=Path, required=True)

    args = parser.parse_args()
    build_all_v_all_matrix(args.phage_dir, args.host_dir, args.output_path)


if __name__ == "__main__":
    main()
