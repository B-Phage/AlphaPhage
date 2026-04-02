"""Utilities for working with our custom MSA files.

These functions are independent and do not require importing the entire HMM package.
They provide table-based access to MSAs and conversion to standard FASTA format.
"""

from pathlib import Path

import pandas as pd


def read_msa_table(msa_file: Path) -> pd.DataFrame:
    """Read a custom MSA file into a pandas DataFrame.

    Each line of the MSA file represents one sequence, with the format:

        SEQUENCE,organism_name;protein_name

    The returned DataFrame has the following columns:

        - sequence: str, the aligned amino acid sequence
        - organism_name: str, organism or source of the protein
        - protein_name: str, the protein identifier

    Args:
        msa_file (Path): Path to the custom MSA file.

    Returns:
        pd.DataFrame: DataFrame containing columns `sequence`, `organism_name`, and `protein_name`.
    """
    df = pd.read_csv(
        msa_file,
        header=None,  # no header in the file
        names=["sequence", "organism_name;protein_name"],  # assign column names
    )

    df[["organism_name", "protein_name"]] = df["organism_name;protein_name"].str.split(
        ";", expand=True
    )
    return df[["sequence", "organism_name", "protein_name"]]


def export_msa_as_fasta(df: pd.DataFrame, out_path: Path) -> None:
    """Export a table-style MSA DataFrame to a standard FASTA file.

    The DataFrame must contain the columns:

        - sequence
        - organism_name
        - protein_name

    Each sequence is written in FASTA format with the header:

        >organism_name;protein_name

    Args:
        df (pd.DataFrame): DataFrame containing the MSA sequences.
        out_path (Path): Path to the output FASTA file.
    """
    with open(out_path, "w") as f:
        for _, row in df.iterrows():
            f.write(f">{row['organism_name']};{row['protein_name']}\n{row['sequence']}\n")
