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
import os
from pathlib import Path
from typing import Iterable, cast

import pyhmmer
import utils.consol_log as log
from hmm.annotations import AnnotationProvider
from hmm.duckdb_io import write_hmm_results_with_sequences
from hmm.io import load_hmms_from_file, load_single_hmm_per_file, write_hmm_scan_summary
from hmm.search import run_hmm_search
from hmm.types import Protein

logger = log.get_logger()


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

    # Load HMMs
    path = args.hmm_path
    if not os.path.exists(path):
        logger.error(f"--hmm-path: '{path}' does not exist.")
        raise ValueError("HMMs path does not exist")
    if os.path.isdir(path):
        logger.info(f"Processing directory: {path}")
        # List files in directory
        hmms = list(load_single_hmm_per_file(path, "*.hmm"))
    elif os.path.isfile(path):
        logger.info(f"Processing single hmm file: {path}")
        # Process the individual file
        hmms = list(load_hmms_from_file(path))
    else:
        logger.error(f"--hmm-path: '{path}' is neither a file nor a directory")
        raise ValueError("HMMs path format unknown")

    # Load proteins
    aa = pyhmmer.easel.Alphabet.amino()
    with pyhmmer.easel.SequenceFile(args.in_fasta, digital=True, alphabet=aa) as seqs_file:
        proteins_block = seqs_file.read_block()
        proteins = cast(Iterable[Protein], proteins_block)

    organism_name = os.path.splitext(os.path.basename(args.in_fasta))[0]

    # Load annotation provider
    annotations = AnnotationProvider(args.method, args.annotation_file)

    # Run HMM search
    df_results = run_hmm_search(
        proteins=proteins_block,  # pyright: ignore[reportArgumentType]
        hmms=hmms,
        annotations=annotations,
        organism_name=organism_name,
        evalue_cutoff=args.evalue_cutoff,
    )

    # Write summary output for tracking
    if len(df_results) == 0:
        logger.info("No hits found.")
        with open(args.out_csv, "w") as f:
            f.write("")
    else:
        write_hmm_scan_summary(df_results, args.out_csv, unique_cols=("hmm_domain_name",))
        # df_results.to_csv(args.out_csv, sep=",", index=False)

    # Write everything to DuckDB
    if args.out_database:
        write_hmm_results_with_sequences(
            df_results,
            proteins,
            organism_name,
            Path(args.out_database),
        )


if __name__ == "__main__":
    main_cli()
