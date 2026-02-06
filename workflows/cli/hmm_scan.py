"""
Command-line entrypoint for running HMM scans.

Responsibilities:
- Parse command-line arguments
- Validate user inputs
- Invoke the HMM package API
- Translate exceptions into process exit codes

This script contains no domain logic.
"""

#!/usr/bin/env python3
import argparse
from pathlib import Path

from hmm.api import run_hmm_scan_on_file


def main_cli():
    parser = argparse.ArgumentParser(description="Run HMM search with annotations.")
    parser.add_argument(
        "-f", "--in-fasta", dest="in_fasta", required=True, type=str, help="Input FASTA file"
    )
    parser.add_argument(
        "-hp",
        "--hmm-path",
        dest="hmm_path",
        required=True,
        help="Path to a single HMM file or a directory containing HMM files",
    )
    parser.add_argument(
        "-a",
        "--annotation-file",
        dest="annotation_file",
        required=True,
        type=str,
        help="HMM annotations file",
    )
    parser.add_argument(
        "-m",
        "--method",
        choices=["phrogs", "pfam"],
        default="phrogs",
        help="Annotation method",
    )
    parser.add_argument(
        "-o",
        "--out-csv",
        dest="out_csv",
        required=True,
        type=str,
        help="Output CSV file for HMM search results",
    )
    parser.add_argument(
        "-db",
        "--out-database",
        dest="out_database",
        required=True,
        type=str,
        help="Parquet lake to add organism results to",
    )
    parser.add_argument(
        "-ec",
        "--evalue-cutoff",
        dest="evalue_cutoff",
        default=1e-5,
        type=int,
        help="Optional E-value cuttoff",
    )
    args = parser.parse_args()

    run_hmm_scan_on_file(
        in_fasta=Path(args.in_fasta),
        hmm_path=Path(args.hmm_path),
        annotation_file=Path(args.annotation_file),
        method=args.method,
        out_csv=Path(args.out_csv),
        out_database=Path(args.out_database),
        evalue_cutoff=args.evalue_cutoff,
    )


if __name__ == "__main__":
    main_cli()
