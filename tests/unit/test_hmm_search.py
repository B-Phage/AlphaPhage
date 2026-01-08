from unittest.mock import MagicMock

import pandas as pd
import pytest
from hmm import annotations as ann
from hmm import search
from hmm.types import HMMHit

# -----------------------------
# Fixtures
# -----------------------------


@pytest.fixture
def mock_proteins():
    """Dummy DigitalSequenceBlock"""
    mock = MagicMock()
    mock.__len__.return_value = 2
    return mock


@pytest.fixture
def mock_hmm():
    """Dummy HMM model"""
    mock = MagicMock()
    mock.name = b"PF00001"
    return [mock]


@pytest.fixture
def mock_annotation_provider():
    """Mock AnnotationProvider"""
    provider = MagicMock(spec=ann.AnnotationProvider)
    provider.get.side_effect = lambda x: f"annotation_{x}"
    return provider


# -----------------------------
# Tests
# -----------------------------


def test_run_hmm_search_basic(monkeypatch, mock_proteins, mock_hmm, mock_annotation_provider):
    """
    Test that a simple search produces a dataframe with expected columns and values.
    """
    # Mock a hit object
    mock_hit = MagicMock()
    mock_hit.included = True
    mock_hit.name = b"protA"
    mock_hit.score = 50
    mock_hit.accession = "ACC123"
    mock_hit.bias = 0.1
    mock_hit.description = "Some description"
    mock_hit.dropped = False
    mock_hit.duplicate = False
    mock_hit.evalue = 1e-10
    mock_hit.hits = 1
    mock_hit.new = True
    mock_hit.best_domain.env_from = 5
    mock_hit.best_domain.env_to = 50
    mock_hit.best_domain.score = 45
    mock_hit.best_domain.alignment.hmm_from = 1
    mock_hit.best_domain.alignment.hmm_to = 45
    mock_hit.best_domain.alignment.hmm_length = 45
    mock_hit.pre_score = 40
    mock_hit.pvalue = 0.001
    mock_hit.reported = True
    mock_hit.sum_score = 95

    # Mock tophits returned by pyhmmer.hmmsearch
    mock_tophit = MagicMock()
    mock_tophit.query.name = b"PF00001"
    mock_tophit.__iter__.return_value = [mock_hit]

    # Patch pyhmmer.hmmsearch to return our mocked results
    monkeypatch.setattr(search.pyhmmer, "hmmsearch", lambda hmms, prots, E: [mock_tophit])

    df = search.run_hmm_search(
        proteins=mock_proteins,
        hmms=mock_hmm,
        annotations=mock_annotation_provider,
        organism_name="Ecoli",
        evalue_cutoff=1e-5,
    )

    # Basic assertions
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "protein_name" in df.columns
    assert "hmm_domain_name" in df.columns
    assert df["protein_name"].iloc[0] == "protA"
    assert df["hmm_domain_name"].iloc[0] == "PF00001"
    assert df["annot"].iloc[0] == "annotation_PF00001"
    assert df["organism_name"].iloc[0] == "Ecoli"

    # Check multi-domain columns exist
    assert "hmm_domain_hit_count_on_prot" in df.columns
    assert "prot_hit_count_for_hmm_dom" in df.columns
    assert "top_hmm_domain_hit_on_prot" in df.columns
    assert "top_protein_match_for_hmm" in df.columns


def test_run_hmm_search_no_hits(monkeypatch, mock_proteins, mock_hmm, mock_annotation_provider):
    """
    Test behavior when hmmsearch returns no included hits.
    """
    mock_tophit = MagicMock()
    mock_tophit.query.name = b"PF00002"
    mock_tophit.__iter__.return_value = []

    monkeypatch.setattr(search.pyhmmer, "hmmsearch", lambda hmms, prots, E: [mock_tophit])

    df = search.run_hmm_search(
        proteins=mock_proteins,
        hmms=mock_hmm,
        annotations=mock_annotation_provider,
        organism_name="Ecoli",
        evalue_cutoff=1e-5,
    )

    # Should return an empty DataFrame
    assert isinstance(df, pd.DataFrame)
    assert df.empty


def test_run_hmm_search_filters_excluded_hits(
    monkeypatch, mock_proteins, mock_hmm, mock_annotation_provider
):
    """
    Test that hits with included=False are skipped.
    """
    mock_hit = MagicMock()
    mock_hit.included = False  # should be skipped
    mock_hit.name = b"protB"
    mock_tophit = MagicMock()
    mock_tophit.query.name = b"PF00003"
    mock_tophit.__iter__.return_value = [mock_hit]

    monkeypatch.setattr(search.pyhmmer, "hmmsearch", lambda hmms, prots, E: [mock_tophit])

    df = search.run_hmm_search(
        proteins=mock_proteins,
        hmms=mock_hmm,
        annotations=mock_annotation_provider,
        organism_name="Ecoli",
    )

    # DataFrame should be empty since hit is excluded
    assert df.empty


def test_run_hmm_search_annotations_called(
    monkeypatch, mock_proteins, mock_hmm, mock_annotation_provider
):
    """
    Test that AnnotationProvider.get() is called for each domain name.
    """
    mock_hit = MagicMock()
    mock_hit.included = True
    mock_hit.name = b"protC"
    mock_hit.best_domain.env_from = 1
    mock_hit.best_domain.env_to = 10
    mock_hit.best_domain.score = 5
    mock_hit.best_domain.alignment.hmm_from = 1
    mock_hit.best_domain.alignment.hmm_to = 10
    mock_hit.best_domain.alignment.hmm_length = 10
    mock_hit.score = 1
    mock_hit.accession = "ACC"
    mock_hit.bias = 0
    mock_hit.description = ""
    mock_hit.dropped = False
    mock_hit.duplicate = False
    mock_hit.evalue = 0.1
    mock_hit.hits = 1
    mock_hit.new = True
    mock_hit.pre_score = 0
    mock_hit.pvalue = 0
    mock_hit.reported = True
    mock_hit.sum_score = 0

    mock_tophit = MagicMock()
    mock_tophit.query.name = b"PF00004"
    mock_tophit.__iter__.return_value = [mock_hit]

    monkeypatch.setattr(search.pyhmmer, "hmmsearch", lambda hmms, prots, E: [mock_tophit])

    df = search.run_hmm_search(
        proteins=mock_proteins,
        hmms=mock_hmm,
        annotations=mock_annotation_provider,
        organism_name="Ecoli",
    )

    mock_annotation_provider.get.assert_called_with("PF00004")
    assert not df.empty
