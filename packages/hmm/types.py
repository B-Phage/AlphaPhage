"""
Shared type definitions for the HMM package.

Responsibilities:
- Define dataclasses, TypedDicts, and Protocols
- Provide explicit contracts between modules
- Improve static typing and testability of the HMM pipeline

No runtime logic should live in this module.
"""

from typing import Iterable, Protocol, TypedDict


class Protein(Protocol):
    """
    Minimal interface for a protein sequence object from pyhmmer.easel.
    """

    name: bytes

    def textize(self) -> "ProteinSequence": ...


class ProteinSequence(Protocol):
    """
    Interface for the object returned by protein.textize().
    """

    sequence: str


class HMMHit(TypedDict):
    """
    Representation of a single HMM hit result.
    """

    protein_name: str
    organism_name: str
    hmm_domain_name: str
    annot: str
    evalue: float
    score: float
    bias: float
    length: int
    top_hmm_domain_hit_on_prot: int
    top_protein_match_for_hmm: int
    best_dom_env_start: int
    best_dom_env_end: int
    best_dom_env_length: int
    sequence: str


class AnnotationProviderProtocol(Protocol):
    """
    Protocol for the AnnotationProvider class.
    """

    def get(self, domain_name: str) -> str: ...


class HMMSearchResults(TypedDict, total=False):
    """
    Optional result fields from run_hmm_search.
    """

    protein_name: str
    organism_name: str
    hmm_domain_name: str
    score: float
    evalue: float
    annot: str
    length: int
    top_hmm_domain_hit_on_prot: int
    top_protein_match_for_hmm: int
