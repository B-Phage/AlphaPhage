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


def run_hmm_scan_on_file(
    in_fasta: Path,
    hmm_path: Path,
    annotation_file: Path,
    method: str,
    out_csv: Path,
    out_database: Path,
    evalue_cutoff: float = 1e-5,
) -> None:
    """
    Run an HMM scan on a single protein FASTA file using a single HMM or HMM directory.

    Parameters
    ----------
    in_fasta : Path
        Protein FASTA file to scan.
    hmm_path : Path
        Path to a single HMM file or a directory of HMM files.
    annotation_file : Path
        HMM annotation CSV/TSV file.
    method : str
        Annotation method, e.g., "phrogs" or "pfam".
    out_csv : Path
        Path to write the CSV summary of hits.
    out_database : Path
        Path to write full results to DuckDB/Parquet lake.
    evalue_cutoff : float, optional
        Threshold for reporting hits, by default 1e-5
    """
    # -----------------------------
    # Load HMMs
    # -----------------------------
    path = Path(hmm_path)
    if not path.exists():
        logger.error(f"HMM path does not exist: {hmm_path}")
        raise ValueError("HMM path does not exist")

    if path.is_dir():
        logger.info(f"Processing HMM directory: {hmm_path}")
        hmms = list(load_single_hmm_per_file(path, "*.hmm"))
    else:
        # If it exists and is not a dir, it MUST be a file
        logger.info(f"Processing single HMM file: {hmm_path}")
        hmms = list(load_hmms_from_file(path))

    # -----------------------------
    # Load proteins
    # -----------------------------
    aa = pyhmmer.easel.Alphabet.amino()
    with pyhmmer.easel.SequenceFile(in_fasta, digital=True, alphabet=aa) as seqs_file:
        proteins_block = seqs_file.read_block()
        proteins = cast(Iterable[Protein], proteins_block)

    organism_name = in_fasta.stem

    # -----------------------------
    # Load annotations
    # -----------------------------
    annotations = AnnotationProvider(method, annotation_file)

    # -----------------------------
    # Run HMM search
    # -----------------------------
    df_results = run_hmm_search(
        proteins=proteins_block,  # pyright: ignore[reportArgumentType]
        hmms=hmms,
        annotations=annotations,
        organism_name=organism_name,
        evalue_cutoff=evalue_cutoff,
    )

    # -----------------------------
    # Write summary CSV
    # -----------------------------
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if len(df_results) == 0:
        logger.info("No hits found, writing empty CSV")
        out_csv.write_text("")
    else:
        write_hmm_scan_summary(df_results, out_csv, unique_cols=("hmm_domain_name",))

    # -----------------------------
    # Write to DuckDB / Parquet lake
    # -----------------------------
    if out_database:
        write_hmm_results_with_sequences(
            df_results,
            proteins,
            organism_name,
            Path(out_database),
        )
