import builtins
from pathlib import Path
from unittest.mock import patch

import hmm.annotations as mod
import pandas as pd
import pytest
from hmm.annotations import AnnotationProvider, PfamSource, PhrogSource

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_phrog_file(tmp_path: Path) -> Path:
    """
    Create a minimal PHROGs-like TSV file.
    """
    path = tmp_path / "phrogs.tsv"

    data = [
        {"phrog": "1", "annot": "NA", "category": "unknown function"},
        {"phrog": "2", "annot": "Capsid protein", "category": "unknown function"},
    ]

    # Write to TSV
    pd.DataFrame(data).to_csv(path, sep="\t", index=False)
    return path


@pytest.fixture
def tmp_pfam_file(tmp_path: Path) -> Path:
    """
    Create a minimal Pfam-style HMM file.
    """
    content = (
        "#=GF ID PF00001\n"
        "#=GF DE Some domain description\n"
        "//\n"
        "#=GF ID PF00002\n"
        "//\n"  # no description
    )
    path = tmp_path / "Pfam-A.hmm"
    path.write_text(content)
    return path


# ---------------------------------------------------------------------------
# PhrogSource tests
# ---------------------------------------------------------------------------


def test_phrog_load_to_dict(tmp_phrog_file: Path):
    source = PhrogSource()
    data = source.load_to_dict(tmp_phrog_file)

    assert data == {"1": "-", "2": "Capsid protein"}


def test_phrog_normalize_id():
    source = PhrogSource()

    assert source.normalize_id("phrog_123") == "123"
    assert source.normalize_id("123") == "123"
    assert source.normalize_id("notphrog_123") == "notphrog_123"


# ---------------------------------------------------------------------------
# PfamSource tests
# ---------------------------------------------------------------------------


def test_pfam_load_to_dict(tmp_pfam_file: Path):
    source = PfamSource()
    data = source.load_to_dict(tmp_pfam_file)

    assert data == {
        "PF00001": "Some domain description",
        "PF00002": "-",  # no description
    }


def test_pfam_normalize_id_is_identity():
    source = PfamSource()

    assert source.normalize_id("PF12345") == "PF12345"


# ---------------------------------------------------------------------------
# AnnotationProvider construction
# ---------------------------------------------------------------------------


def test_provider_rejects_unknown_source(tmp_path: Path):
    with pytest.raises(ValueError, match="Unsupported source"):
        AnnotationProvider("unknown", tmp_path / "file.txt")


def test_provider_warns_on_missing_file(tmp_path: Path, caplog):
    missing = tmp_path / "missing.txt"
    AnnotationProvider("phrogs", missing)

    assert any(
        "Annotation file not found at startup" in record.message for record in caplog.records
    )


# ---------------------------------------------------------------------------
# Lazy-loading behavior
# ---------------------------------------------------------------------------


def test_lazy_loading_only_on_first_access(tmp_phrog_file: Path, monkeypatch):
    provider = AnnotationProvider("phrogs", tmp_phrog_file)

    called = {"count": 0}

    original = provider.engine.load_to_dict

    def wrapped(path):
        called["count"] += 1
        return original(path)

    monkeypatch.setattr(provider.engine, "load_to_dict", wrapped)

    # No load on construction
    assert called["count"] == 0

    # First access triggers load
    provider.get("phrog_1")
    assert called["count"] == 1

    # Subsequent access uses cache
    provider.get("phrog_2")
    provider.get("phrog_3")
    assert called["count"] == 1


# ---------------------------------------------------------------------------
# AnnotationProvider.get
# ---------------------------------------------------------------------------


def test_get_returns_annotation(tmp_phrog_file: Path):
    provider = AnnotationProvider("phrogs", tmp_phrog_file)

    assert provider.get("phrog_1") == "-"
    assert provider.get("phrog_2") == "Capsid protein"


def test_get_returns_dash_for_missing_domain(tmp_phrog_file: Path):
    provider = AnnotationProvider("phrogs", tmp_phrog_file)

    assert provider.get("phrog_999") == "-"


def test_get_normalizes_domain_ids(tmp_phrog_file: Path):
    provider = AnnotationProvider("phrogs", tmp_phrog_file)

    assert provider.get("phrog_1") == provider.get("1")


# ---------------------------------------------------------------------------
# Missing file handling at access time
# ---------------------------------------------------------------------------


def test_missing_file_returns_empty_dict(tmp_path: Path, caplog):
    missing = tmp_path / "does_not_exist.tsv"
    provider = AnnotationProvider("phrogs", missing)

    value = provider.get("phrog_1")

    assert value == "-"
    assert any("Cannot load annotations" in record.message for record in caplog.records)


# ---------------------------------------------------------------------------
# Cache behavior
# ---------------------------------------------------------------------------


def test_data_property_returns_same_object(tmp_phrog_file: Path):
    provider = AnnotationProvider("phrogs", tmp_phrog_file)

    first = provider.data
    second = provider.data

    assert first is second


# ---------------------------------------------------------------------------
# Error handling in loaders
# ---------------------------------------------------------------------------


def test_phrog_loader_failure_returns_empty_dict(tmp_path: Path):
    dummy_file = tmp_path / "phrogs.tsv"
    dummy_file.write_text("1\t.\t.\t.\t.\t.\tCapsid\n")

    with patch("hmm.annotations.pd.read_csv") as mock_read:
        mock_read.side_effect = RuntimeError("boom")
        source = mod.PhrogSource()
        print(mod.pd.read_csv)
        data = source.load_to_dict(dummy_file)
        assert data == {}


def test_pfam_loader_failure_returns_empty_dict(monkeypatch, tmp_pfam_file: Path):
    source = PfamSource()

    def broken_open(*args, **kwargs):
        raise IOError("boom")

    monkeypatch.setattr(
        "hmm.annotations.Path.open",
        broken_open,
    )

    data = source.load_to_dict(tmp_pfam_file)
    assert data == {}
