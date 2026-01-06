# tests/test_hmm_data_setup.py
import gzip
import hashlib
import io
import shutil
import tarfile
import tempfile
import zipfile
from pathlib import Path
from unittest.mock import patch

import pytest
import setup_workflows.hmm_data_setup as hms


# ---------------------------
# Fixtures
# ---------------------------
@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def small_text_file(tmp_dir):
    path = tmp_dir / "file.txt"
    path.write_text("hello world")
    return path


@pytest.fixture
def small_gz_file(tmp_dir, small_text_file):
    gz_path = tmp_dir / "file.txt.gz"
    with gzip.open(gz_path, "wb") as f_out, small_text_file.open("rb") as f_in:
        shutil.copyfileobj(f_in, f_out)
    return gz_path


@pytest.fixture
def small_tar_file(tmp_dir, small_text_file):
    tar_path = tmp_dir / "archive.tar"
    with tarfile.open(tar_path, "w") as tar:
        tar.add(small_text_file, arcname="file.txt")
    return tar_path


@pytest.fixture
def small_zip_file(tmp_dir, small_text_file):
    zip_path = tmp_dir / "archive.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.write(small_text_file, arcname="file.txt")
    return zip_path


# ---------------------------
# Utility function tests
# ---------------------------
def test_compute_and_verify_checksum(tmp_dir, small_text_file):
    expected = hashlib.sha256(small_text_file.read_bytes()).hexdigest()
    assert hms.compute_sha256(small_text_file) == expected

    # verify checksum passes
    hms.verify_checksum(small_text_file, expected)

    # verify checksum fails
    with pytest.raises(ValueError):
        hms.verify_checksum(small_text_file, "wronghash")


def test_final_artifact_is_valid(tmp_dir, small_text_file):
    expected_sha = hashlib.sha256(small_text_file.read_bytes()).hexdigest()
    assert hms.final_artifact_is_valid(small_text_file, expected_sha)
    # delete file, should return False
    small_text_file.unlink()
    assert not hms.final_artifact_is_valid(small_text_file, expected_sha)


# ---------------------------
# Extraction tests
# ---------------------------
def test_extract_from_gz(tmp_dir, small_text_file, small_gz_file):
    out_path = tmp_dir / "out.txt"
    hms.extract_from_gz(small_gz_file, out_path)
    assert out_path.read_bytes() == small_text_file.read_bytes()


def test_extract_from_tar(tmp_dir, small_text_file, small_tar_file):
    out_path = tmp_dir / "out.txt"
    hms.extract_from_tar(small_tar_file, "file.txt", out_path)
    assert out_path.read_bytes() == small_text_file.read_bytes()


def test_extract_from_zip(tmp_dir, small_text_file, small_zip_file):
    out_path = tmp_dir / "out.txt"
    hms.extract_from_zip(small_zip_file, "file.txt", out_path)
    assert out_path.read_bytes() == small_text_file.read_bytes()


def test_materialize_hmm_artifact(tmp_dir, small_text_file, small_gz_file):
    dest = tmp_dir / "artifact.txt"

    # raw
    hms.materialize_hmm_artifact(small_text_file, dest, "raw")
    assert dest.read_bytes() == small_text_file.read_bytes()

    # gz
    hms.materialize_hmm_artifact(small_gz_file, dest, "gz")
    assert dest.read_bytes() == small_text_file.read_bytes()

    # tar
    tar_path = tmp_dir / "archive.tar"
    with tarfile.open(tar_path, "w") as tar:
        tar.add(small_text_file, arcname="file.txt")
    hms.materialize_hmm_artifact(tar_path, dest, "tar", inner_path="file.txt")
    assert dest.read_bytes() == small_text_file.read_bytes()

    # zip
    zip_path = tmp_dir / "archive.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.write(small_text_file, arcname="file.txt")
    hms.materialize_hmm_artifact(zip_path, dest, "zip", inner_path="file.txt")
    assert dest.read_bytes() == small_text_file.read_bytes()


# ---------------------------
# Core workflow tests
# ---------------------------
@patch("setup_workflows.hmm_data_setup.split_hmms")
@patch("setup_workflows.hmm_data_setup.download_file")
def test_fetch_and_split_hmm(mock_download, mock_split, tmp_dir, small_text_file):
    def fake_download(url, dest_path):
        shutil.copyfile(small_text_file, dest_path)
        return dest_path

    mock_download.side_effect = fake_download

    final_sha256 = hms.compute_sha256(small_text_file)

    result = hms.fetch_and_split_hmm(
        hmm_name="test",
        source_url="http://example.com/file",
        dest_dir=tmp_dir,
        final_sha256=final_sha256,
        source_format="raw",
        can_split=True,
    )

    assert result.exists()
    mock_split.assert_called_once()


# ---------------------------
# YAML registry tests
# ---------------------------
@patch("setup_workflows.hmm_data_setup.fetch_and_split_hmm")
def test_fetch_from_registry(mock_fetch, tmp_dir):
    yaml_file = tmp_dir / "registry.yaml"
    yaml_file.write_text(
        """
- name: test1
  source_url: http://example.com/file1
  dest_dir: tmp_dir
  final_sha256: abc123
  source_format: raw
  can_split: true
- name: test2
  source_url: http://example.com/file2
  dest_dir: tmp_dir
  final_sha256: def456
  source_format: gz
  can_split: false
"""
    )
    hms.fetch_from_registry(yaml_file)
    assert mock_fetch.call_count == 2


# ---------------------------
# CLI tests
# ---------------------------
@patch("setup_workflows.hmm_data_setup.fetch_from_registry")
@patch("setup_workflows.hmm_data_setup.fetch_and_split_hmm")
def test_cli_yaml(mock_single, mock_registry, tmp_dir, monkeypatch):
    # Simulate CLI with YAML
    yaml_file = tmp_dir / "registry.yaml"
    yaml_file.write_text("[]")
    monkeypatch.setattr("sys.argv", ["prog", "--yaml-registry", str(yaml_file)])
    hms.main()
    mock_registry.assert_called_once_with(yaml_file)


@patch("setup_workflows.hmm_data_setup.fetch_and_split_hmm")
def test_cli_single(mock_single, tmp_dir, monkeypatch):
    final_sha256 = "abc"
    monkeypatch.setattr(
        "sys.argv",
        [
            "prog",
            "--hmm-name",
            "test",
            "--source-url",
            "http://example.com/file",
            "--dest-dir",
            str(tmp_dir),
            "--final-sha256",
            final_sha256,
        ],
    )
    with patch("setup_workflows.hmm_data_setup.final_artifact_is_valid", return_value=False):
        with patch("setup_workflows.hmm_data_setup.download_file") as mock_dl:
            mock_dl.return_value = tmp_dir / "test"
            hms.main()
            mock_single.assert_called()


# ---------------------------
# Dwonload tests
# ---------------------------
def test_download_file_downloads_when_missing(tmp_path: Path):
    dest = tmp_path / "downloaded.txt"
    fake_content = b"hello hmm\n"

    fake_response = io.BytesIO(fake_content)

    with patch("setup_workflows.hmm_data_setup.urllib.request.urlopen") as mock_urlopen:
        mock_urlopen.return_value.__enter__.return_value = fake_response

        result = hms.download_file(
            url="http://example.com/file.txt",
            dest_path=dest,
        )

    assert result == dest
    assert dest.exists()
    assert dest.read_bytes() == fake_content
    mock_urlopen.assert_called_once()


def test_download_file_skips_if_exists(tmp_path: Path):
    dest = tmp_path / "already_here.txt"
    original_content = b"existing file\n"
    dest.write_bytes(original_content)

    with patch("setup_workflows.hmm_data_setup.urllib.request.urlopen") as mock_urlopen:
        result = hms.download_file(
            url="http://example.com/file.txt",
            dest_path=dest,
        )

    assert result == dest
    assert dest.read_bytes() == original_content
    mock_urlopen.assert_not_called()
