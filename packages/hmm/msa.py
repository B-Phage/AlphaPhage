# msa construction
from pathlib import Path
from typing import Iterable, Protocol, cast

import pyhmmer
from hmm.io import to_str
from utils import consol_log as log

logger = log.get_logger()


class _MSAProtocol(Protocol):
    # Pylance does not believe pyhmmer.easel.MSA has an alignment attribute at all
    # We use this to cast to what we expect at runtime
    names: Iterable[bytes]
    alignment: Iterable[str]


def make_msa(hmm_seq_hits_file: Path, hmm_file_path: Path, out_msa_path: Path) -> Path:
    """
    Create a multiple sequence alignment (MSA) from sequences matching an HMM,
    and write it in a table-like format for easy pandas ingestion.

    Each line of the output file represents one sequence, with the format:

        SEQUENCE,organism_name;protein_name

    This is intended for downstream clustering using pandas and scikit-learn.

    Parameters
    ----------
    hmm_seq_hits_file : Path
        FASTA file containing sequences that matched the HMM.
    hmm_file_path : Path
        Path to the HMM file (hmmer3 format).
    out_msa_path : Path
        Path to output MSA file (.csa expected). Each line will be formatted
        as: SEQUENCE,organism_name;protein_name

    Returns
    -------
    Path
        Path to the MSA output file.

    Notes
    -----
    - Does not produce standard FASTA-format MSA.
    - Intended for table-based clustering pipelines.
    """
    if out_msa_path.suffix != ".csa":
        raise ValueError(f'Expecting ".csa" output file, got "{out_msa_path.suffix}"')

    out_msa_path.parent.mkdir(parents=True, exist_ok=True)

    # Read HMM
    with pyhmmer.plan7.HMMFile(hmm_file_path) as hmm_file:
        hmm = hmm_file.read()

    # Ensure we actually got a HMM
    assert hmm is not None

    hmm_name = to_str(hmm.name)
    logger.info(f"Read HMM: {hmm_name}")

    # Read sequences
    with pyhmmer.easel.SequenceFile(hmm_seq_hits_file, digital=True) as seqs_file:
        proteins = seqs_file.read_block()
    logger.info(f"Read {len(proteins)} sequences from {hmm_seq_hits_file}")

    # Perform MSA
    try:
        msa_raw = pyhmmer.hmmer.hmmalign(
            hmm, proteins, trim=False, digitize=False, all_consensus_cols=True
        )
    except Exception as e:
        logger.error(f"Error aligning sequences to HMM {hmm_name}: {e}")
        raise
    msa = cast(_MSAProtocol, msa_raw)

    # Write table-style MSA: sequence,organism_name;protein_name
    with open(out_msa_path, "w") as out_file:
        for name_bytes, sequence in zip(msa.names, msa.alignment):
            header = to_str(name_bytes)  # protein_name or combined header
            # keep organism_name;protein_name if available in the header
            out_file.write(f"{sequence},{header}\n")

    logger.info(f"MSA table written to {out_msa_path}")
    return out_msa_path
