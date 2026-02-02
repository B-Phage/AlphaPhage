"""
Input/output utilities for HMM workflows.

Responsibilities:
- Load HMMs from local files or directories
- Handle HMM file formats and pressed databases
- Provide lightweight context managers for HMM file iteration

No domain logic or search logic is implemented here.
"""

import glob
import os
from pathlib import Path
from typing import Union

import pandas as pd
import pyhmmer
import pyhmmer.plan7
import utils.consol_log as log

logger = log.get_logger(level=log.INFO)


def load_single_hmm_per_file(hmm_dir, patt):
    """Load the first HMM from an HMM file located on the local filesystem: (hmmer3 format).
    Example patt: "*.hhm"
    """
    hmms = []
    lst_hmm_files = glob.glob(hmm_dir + "/" + patt)
    if lst_hmm_files == []:
        logger.error(f"No hmms found at {hmm_dir} with pattern {patt}. Exiting.")
        raise ValueError("No HMMs found")
    for f in lst_hmm_files:
        # Note: Needs to be hmmer3 format - phrogs format does not work
        with pyhmmer.plan7.HMMFile(f) as hmm_file:
            hmm = hmm_file.read()
            hmms.append(hmm)
    return hmms


def load_hmms_from_file(hmm_multifile):
    """Load the all HMMs from an HMM file located on the local filesystem: (hmmer3 format).
    Example patt: "*.hhm"
    """
    with pyhmmer.plan7.HMMFile(hmm_multifile) as hmm_file:
        hmms = list(hmm_file)
    return hmms


def split_hmms(hmm_multifile: Union[str, Path], out_dir: Union[str, Path]) -> None:
    """
    Load all HMMs from a single HMM file (hmmer3 format) and split them into individual files.

    Parameters
    ----------
    hmm_multifile : str | Path
        Path to the HMM file containing multiple HMMs.
    out_dir : str | Path
        Directory where individual HMM files will be written.
    """
    hmms = load_hmms_from_file(hmm_multifile)  # get list of hmms
    os.makedirs(out_dir, exist_ok=True)
    for i, hmm in enumerate(hmms):
        # Each HMM has a 'name' attribute (bytes), and optionally an 'accession' (bytes).
        # It's good practice to use the accession if available, otherwise the name.
        # We need to decode them to string for filenames.

        filename_base = None
        if hmm.name:  # name prefered
            filename_base = to_str(hmm.name)
        elif hmm.accession:
            filename_base = to_str(hmm.accession)
        else:
            # Fallback if neither name nor accession is present (unlikely for well-formed HMMs)
            filename_base = f"unnamed_hmm_{i}"
            logger.warning(f"HMM name not found, writing as {filename_base}.hmm")

        output_filepath = os.path.join(out_dir, f"{filename_base}.hmm")

        try:
            with open(output_filepath, "wb") as output_file:  # Open in binary write mode
                hmm.write(output_file)
            logger.debug(f"  Wrote {filename_base}.hmm")
        except Exception as e:
            logger.error(f" could not write HMM {filename_base}.hmm: {e}")

    logger.info(f"Done writing individual HMM files from {hmm_multifile}")


def load_pressed_hmms(hmm_pressed_file):
    """Get an iterator over the OptimizedProfile in the HMM database.
    A call to find hits like below given hmms as optimized_profiles would be faster than when the models are an iterator (e.g. list) of hmms which has to be converted from file.
    hits = next(pyhmmer.hmmsearch(hmms, proteins))
    #"""
    with pyhmmer.plan7.HMMFile(hmm_pressed_file) as hmm_file:
        if not hmm_file.is_pressed():
            logger.error(
                "Error: hmm_file " + hmm_pressed_file + " is not in the pressed format. Exiting."
            )
            raise ValueError("No HMMs found")
        else:
            hmms = hmm_file.optimized_profiles()
            return hmms


def write_hmm_scan_summary(
    df: pd.DataFrame,
    out_csv: str | Path,
    unique_cols: tuple[str, ...] = (),
    max_unique_preview: int = 20,
) -> None:
    """
    Writes a tiny CSV summary suitable for pipeline tracking.
    """

    summary: dict[str, Union[int, str]] = {
        "n_rows": df.shape[0],
        "n_columns": df.shape[1],
    }

    for col in unique_cols:
        s = df[col]
        nunique = s.nunique(dropna=True)
        summary[f"{col}_nunique"] = nunique

        # Preview only if small
        if nunique <= max_unique_preview:
            summary[f"{col}_unique_preview"] = ",".join(map(str, s.dropna().unique()))
        else:
            summary[f"{col}_unique_preview"] = f">{max_unique_preview}"

    pd.DataFrame([summary]).to_csv(out_csv, index=False)


def to_str(value):
    """Convert bytes to string if needed, else return as-is"""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value
