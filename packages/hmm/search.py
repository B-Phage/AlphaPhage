"""
Core HMM search logic.

Responsibilities:
- Run HMMER searches using pyhmmer
- Iterate over hits and extract relevant statistics
- Attach annotations to HMM hits
- Return structured search results as tabular data

This module is pure computation and contains no persistence or CLI logic.
"""

from typing import Iterable, Union

import pandas as pd
import pyhmmer
import utils.consol_log as log
from hmm.annotations import AnnotationProvider
from pyhmmer.easel import DigitalSequenceBlock
from pyhmmer.plan7 import HMM

logger = log.get_logger()


def normalize_name(name: str | bytes | bytearray | memoryview) -> str:
    """Normalize pyhmmer name fields to a Python string."""
    if isinstance(name, memoryview):
        return name.tobytes().decode()
    if isinstance(name, (bytes, bytearray)):
        return name.decode()
    return str(name)


def run_hmm_search(
    proteins: DigitalSequenceBlock,
    hmms: Iterable[HMM],
    annotations: "AnnotationProvider",
    organism_name: str,
    evalue_cutoff: float = 1e-5,
) -> pd.DataFrame:
    """
    Run HMMER search against a set of proteins and annotate the hits.

    Parameters
    ----------
    proteins : DigitalSequenceBlock
        Proteins to search against.
    hmms : Iterable[HMM]
        HMM models.
    annotations : AnnotationProvider
        Annotation provider instance (phrogs, pfam, etc.).
    organism_name : str
        Name of the organism (used in results).
    evalue_cutoff : float
        E-value threshold for filtering hits.

    Returns
    -------
    pd.DataFrame
        Annotated HMM hits.
    """

    results = []

    hmms = list(hmms)  # ensure it’s sized
    logger.info(f"Running HMM search: {len(hmms)} HMMs x {len(proteins)} proteins")
    tophits = list(pyhmmer.hmmsearch(hmms, proteins, E=evalue_cutoff))

    for hits in tophits:
        domain_name = normalize_name(hits.query.name)
        for hit in hits:
            if not hit.included:
                continue

            query_prot = normalize_name(hit.name)
            annot = annotations.get(domain_name)

            results.append(
                {
                    "protein_name": query_prot,  # "query_prot" in RnD
                    "organism_name": organism_name,  # "query_genome" in RnD
                    "hmm_domain_name": domain_name,  # "domain" in RnD
                    "score": hit.score,
                    "bias": hit.bias,
                    "evalue": hit.evalue,
                    "length": hit.length,
                    "name": hit.name,
                    "best_dom_env_start": hit.best_domain.env_from,
                    "best_dom_env_end": hit.best_domain.env_to,
                    "best_dom_env_length": hit.best_domain.env_to - hit.best_domain.env_from,
                    "best_dom_score": hit.best_domain.score,
                    "best_dom_ali_start": hit.best_domain.alignment.hmm_from,
                    "best_dom_ali_end": hit.best_domain.alignment.hmm_to,
                    "best_dom_ali_length": hit.best_domain.alignment.hmm_length,
                    "pre_score": hit.pre_score,
                    "pvalue": hit.pvalue,
                    "sum_score": hit.sum_score,
                    "annot": annot,
                }
            )

    if not results:
        return pd.DataFrame()  # empty DataFrame if no hits

    df_results = pd.DataFrame(results)

    # Add multi-domain info
    domain_counts = df_results[
        "protein_name"
    ].value_counts()  # number of duplicate hmm domain hits on a specific protein
    df_results["hmm_domain_hit_count_on_prot"] = df_results["protein_name"].map(
        domain_counts
    )  # was 'multi_domain_hits' in RnD

    domain_counts = df_results[
        "hmm_domain_name"
    ].value_counts()  # number of duplicate protein hits for a specific hmm domain
    df_results["prot_hit_count_for_hmm_dom"] = df_results["hmm_domain_name"].map(
        domain_counts
    )  # was 'domain_repeats' in RnD

    # Rank by score for domains and genes
    df_results["top_hmm_domain_hit_on_prot"] = df_results.groupby(  # was 'top_domain_hit' in RnD
        "protein_name"
    )["score"].rank(method="first", ascending=False)
    df_results["top_protein_match_for_hmm"] = df_results.groupby(  # was 'top_gene_hit' in RnD
        "hmm_domain_name"
    )["score"].rank(method="first", ascending=False)

    return df_results
