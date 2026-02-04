"""
This test set covers both `hmm.msa.py` and `utils.msa_io.py`.

These two modules can be used independently, i.e. you do not need to import the `hmm` package to use `msa_io.py`.
However, if the syntax of one changes, the other may also need updating.

For this reason, the tests are intentionally collocated for ease of maintenance.
"""

from pathlib import Path

import pandas as pd
import pytest
from hmm.msa import make_msa
from utils.msa_io import export_msa_as_fasta, read_msa_table


# -----------------------
# Fixtures
# -----------------------
@pytest.fixture
def phrog_126_hmm():
    return Path("tests/fixtures/phrog_126.hmm")


@pytest.fixture
def phrog_126_fa():
    return Path("tests/fixtures/phrog_126.fa")


@pytest.fixture
def sample_msa_df():
    return pd.DataFrame(
        {
            "sequence": ["MKT-", "AGH-"],
            "organism_name": ["Org1", "Org2"],
            "protein_name": ["Prot1", "Prot2"],
        }
    )


# -----------------------
# Parameterized tests
# -----------------------
@pytest.mark.parametrize(
    "hmm_path, fasta_path",
    [
        (
            Path("tests/fixtures/phrog_126.hmm"),
            Path("tests/fixtures/phrog_126.fa"),
        ),
        (
            Path("tests/fixtures/DUF1244.hmm"),
            Path("tests/fixtures/DUF1244.fa"),
        ),
    ],
)
def test_make_msa_table_format(tmp_path: Path, hmm_path: Path, fasta_path: Path):
    """
    Integration-style test using real HMM and FASTA fixtures.

    Verifies:
    - MSA table file is created
    - File is non-empty
    - Output matches the custom table format: SEQUENCE,organism_name;protein_name
    - At least one sequence is present
    """
    out_file = tmp_path / "out.csa"

    result_path = make_msa(
        hmm_seq_hits_file=fasta_path,
        hmm_file_path=hmm_path,
        out_msa_path=out_file,
    )

    # File exists
    assert result_path.exists(), "MSA file was not created"

    # File is not empty
    contents = result_path.read_text().strip()
    assert contents != "", "MSA file is empty"

    # Validate table format: one comma per line separating sequence and header
    lines = contents.splitlines()
    assert all("," in line for line in lines), "Each line must contain a comma separator"

    # Validate header part (organism_name;protein_name)
    for line in lines:
        sequence, header = line.split(",", 1)
        assert sequence.isalpha() or "-" in sequence, "Sequence contains invalid characters"
        assert ";" in header, "Header must contain organism_name;protein_name"
        org, prot = header.split(";", 1)
        assert org and prot, "Both organism_name and protein_name must be non-empty"

    # Optional: read into pandas to validate columns
    df = read_msa_table(result_path)
    assert not df.empty, "DataFrame created from MSA table is empty"
    assert set(df.columns) == {"sequence", "organism_name", "protein_name"}
    assert all(len(seq) > 0 for seq in df["sequence"])


def test_make_msa_wrong_extension(tmp_path: Path, phrog_126_hmm: Path, phrog_126_fa: Path):
    """
    Check that make_msa raises ValueError if output file does not have '.csa' suffix
    """
    out_file = tmp_path / "out.txt"  # deliberately wrong extension
    import pytest
    from hmm.msa import make_msa

    with pytest.raises(ValueError, match=r"Expecting \".csa\" output file"):
        make_msa(
            hmm_seq_hits_file=phrog_126_fa,
            hmm_file_path=phrog_126_hmm,
            out_msa_path=out_file,
        )


def test_make_msa_empty_hmm(tmp_path: Path, phrog_126_fa: Path):
    """
    Check that make_msa raises ValueError if no HMM is found in the file.
    """
    empty_hmm = tmp_path / "empty.hmm"
    empty_hmm.touch()  # create empty file
    out_file = tmp_path / "out.csa"
    import pytest
    from hmm.msa import make_msa

    with pytest.raises(EOFError, match=r"HMM file is empty"):
        make_msa(
            hmm_seq_hits_file=phrog_126_fa,
            hmm_file_path=empty_hmm,
            out_msa_path=out_file,
        )


def test_make_msa_hmmalign_raises(
    tmp_path: Path, phrog_126_hmm: Path, phrog_126_fa: Path, monkeypatch
):
    """
    Check that make_msa logs and re-raises an exception from hmmalign.
    """
    import pyhmmer.hmmer
    import pytest
    from hmm.msa import make_msa

    out_file = tmp_path / "out.csa"

    # Monkeypatch hmmalign to raise
    def fake_hmmalign(*args, **kwargs):
        raise RuntimeError("forced alignment error")

    monkeypatch.setattr(pyhmmer.hmmer, "hmmalign", fake_hmmalign)

    with pytest.raises(RuntimeError, match="forced alignment error"):
        make_msa(
            hmm_seq_hits_file=phrog_126_fa,
            hmm_file_path=phrog_126_hmm,
            out_msa_path=out_file,
        )


def test_export_msa_as_fasta(tmp_path: Path, sample_msa_df):
    out_path = tmp_path / "out.fasta"
    export_msa_as_fasta(sample_msa_df, out_path)

    assert out_path.exists()
    contents = out_path.read_text().splitlines()
    # Check header lines
    assert contents[0] == ">Org1;Prot1"
    assert contents[2] == ">Org2;Prot2"
    # Check sequence lines
    assert contents[1] == "MKT-"
    assert contents[3] == "AGH-"
