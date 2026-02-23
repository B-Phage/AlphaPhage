from typing import Any, List, Tuple

import pandas as pd

# Allowed amino acid characters + gap
VALID_AA_CHARS = set("ACDEFGHIKLMNPQRSTVWY")  # standard amino acids
VALID_AMBIGUOUS_AA = set("X")  # ambiguous residues
VALID_GAPS = set("-")  # MSA gaps
VALID_STOPS = set("*")  # stop codons (optional)
VALID_FILLERS = set("OX")  # safe fillers, not real amino acids


def standardize_frame_chars(
    df: pd.DataFrame,
    filler_char: str = "-",
    replace_stop_codons: bool = True,
    keep_stop_metadata: bool = True,
) -> Tuple[pd.DataFrame, List[int]]:
    """
    Standardize characters in a dataframe representing aligned sequences.

    Converts all letters to uppercase, replaces '.' with '-', optionally replaces
    '*' with a safe filler character, and validates characters.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe (rows = sequences, columns = positions).
    filler_char : str
        Character to replace stop codons '*' with. Must not be a valid AA or ambiguous AA.
    replace_stop_codons : bool
        Whether to replace '*' with filler_char.
    keep_stop_metadata : bool
        Whether to return a list marking rows that contained '*' originally.

    Returns
    -------
    df_clean : pd.DataFrame
        Dataframe with standardized characters.
    lst_has_stop : List[int]
        List marking which rows originally contained '*'.
    """
    # Ensure df is a DataFrame
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pd.DataFrame, got {type(df)}")

    # Validate filler_char
    all_invalid_fillers = VALID_AA_CHARS | VALID_AMBIGUOUS_AA
    if filler_char in all_invalid_fillers:
        raise ValueError(
            f"filler_char '{filler_char}' cannot be a valid amino acid or ambiguous AA"
        )

    # Track which rows originally had stop codons
    lst_has_stop = (
        [int("*" in row.values) for _, row in df.iterrows()]
        if keep_stop_metadata
        else [0] * len(df)
    )

    # Function to clean a single character
    def _clean_char(e: Any) -> str:
        s = str(e).upper()
        if s == ".":
            s = "-"  # convert dot to gap
        if replace_stop_codons and s == "*":
            s = filler_char
        return s

    # Apply cleaning
    df_clean = df.apply(lambda row: row.map(_clean_char))

    # Determine allowed characters after cleaning
    allowed_chars = VALID_AA_CHARS | VALID_AMBIGUOUS_AA | VALID_GAPS | set(filler_char)
    if not replace_stop_codons:
        allowed_chars |= VALID_STOPS  # allow '*' if not replaced

    # Validate cleaned dataframe
    unique_chars = set(df_clean.values.flatten())
    invalid_chars = unique_chars - allowed_chars
    if invalid_chars:
        raise ValueError(f"Invalid characters found in AA dataframe: {invalid_chars}")

    return df_clean, lst_has_stop


def split_names_diversity(df: pd.DataFrame) -> List[List[str]]:
    """
    Identify consecutive columns in a dataframe that are 'informative' (have >= 2 unique values).

    Parameters
    ----------
    df : pd.DataFrame
        Standardized dataframe (e.g., output from standardize_frame_chars), where each cell is a character.

    Returns
    -------
    List[List[str]]
        A list of lists, each inner list contains consecutive informative column names.
    """
    informative_mask = _get_informative_columns(df)
    return _group_consecutive_columns(df.columns.tolist(), informative_mask)


def _get_informative_columns(df: pd.DataFrame) -> List[bool]:
    """
    Return a boolean mask indicating which columns have >= 2 unique values.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe of characters.

    Returns
    -------
    List[bool]
        True for columns with >= 2 unique values, False otherwise.
    """
    return [(len(df[col].unique()) >= 2) for col in df.columns]


def _group_consecutive_columns(columns: List[str], mask: List[bool]) -> List[List[str]]:
    """
    Group consecutive columns where the mask is True.

    Parameters
    ----------
    columns : List[str]
        Column names.
    mask : List[bool]
        Boolean mask for informative columns.

    Returns
    -------
    List[List[str]]
        Groups of consecutive informative columns.
    """
    grouped: List[List[str]] = []
    current_group: List[str] = []

    for col, is_informative in zip(columns, mask):
        if is_informative:
            current_group.append(col)
        else:
            if current_group:
                grouped.append(current_group)
                current_group = []

    # Add any remaining group at the end
    if current_group:
        grouped.append(current_group)

    return grouped
