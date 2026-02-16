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
    """Loads the first HMM profile from every file matching a pattern in a directory.

    Iterates through a directory, identifies files matching the glob pattern,
    and extracts only the first HMM profile from each file. This is useful
    when HMMs are stored individually rather than in a multi-HMM database.

    Args:
        hmm_dir (str): Path to the directory containing HMM files.
        patt (str): Glob pattern to match files (e.g., "*.hmm").

    Returns:
        list[pyhmmer.plan7.HMM]: A list of HMM objects, one from each matched file.

    Raises:
        ValueError: If no files are found matching the pattern in the directory.
        EOFError: If a matched file is empty or improperly formatted.

    Warning:
        Input files must be in HMMER3 format. Other formats, such as
        PHROGs-specific HMM formats, may not be compatible with ``hmm_file.read()``.

    Example:
        >>> hmms = load_single_hmm_per_file("./models", "*.hmm")
        >>> print(f"Collected {len(hmms)} individual models.")
    """
    hmms = []
    # Using os.path.join is generally safer for cross-platform compatibility
    lst_hmm_files = glob.glob(hmm_dir + "/" + patt)
    if not lst_hmm_files:
        logger.error(f"No hmms found at {hmm_dir} with pattern {patt}. Exiting.")
        raise ValueError("No HMMs found")
    for f in lst_hmm_files:
        # .read() gets the next (in this case, first) HMM in the file
        with pyhmmer.plan7.HMMFile(f) as hmm_file:
            hmm = hmm_file.read()
            hmms.append(hmm)
    return hmms


def load_hmms_from_file(hmm_multifile):
    """Loads all HMMs from a standard HMMER3 file into memory.

    Reads a HMMER3 format file (typically with .hmm extensions) and parses
    all contained profiles into a list of HMM objects.

    Args:
        hmm_multifile (str): Path to the HMM file on the local filesystem.

    Returns:
        list[pyhmmer.plan7.HMM]: A list containing all HMM profiles
            parsed from the file.

    Note:
        Unlike the pressed file loader, this function reads the entire
        contents into memory as a list. For very large databases,
        consider using the iterator-based approach or pressed formats
        to save memory.

    Example:
        >>> hmms = load_hmms_from_file("profiles.hmm")
        >>> print(f"Loaded {len(hmms)} profiles.")
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
    """Loads optimized HMM profiles from a pressed HMM file.

    Retrieves an iterator over the `OptimizedProfile` objects within a database.
    Using pressed files and optimized profiles is significantly faster for
    searches (e.g., ``pyhmmer.hmmsearch``) compared to reading standard HMM
    files, as it avoids on-the-fly conversion.

    Args:
        hmm_pressed_file (str): Path to the pressed HMMER3 database file.

    Returns:
        pyhmmer.plan7.OptimizedProfiles: An iterator over the optimized
            HMM profiles in the database.

    Raises:
        ValueError: If the provided file is not in the pressed format.
        FileNotFoundError: If the specified file path does not exist.

    Example:
        >>> hmms = load_pressed_hmms("path/to/database.h3m")
        >>> hits = next(pyhmmer.hmmsearch(hmms, proteins))
    """
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
