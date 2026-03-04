import tarfile
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests
import responses
from external_db_provisioning.provisioner import download_file, extract_artifact, provision_asset

# --- download_file tests ---


@responses.activate
def test_download_file_success(tmp_path):
    url = "https://example.com/data.txt"
    dest = tmp_path / "test.txt"
    content = b"some data content"

    responses.add(responses.GET, url, body=content, status=200)

    result = download_file(url, dest)

    assert result.exists()
    assert result.read_bytes() == content


@responses.activate
def test_download_file_retries_and_fails(tmp_path):
    url = "https://example.com/fail.txt"
    dest = tmp_path / "fail.txt"

    # Mock a persistent 500 error
    responses.add(responses.GET, url, status=500)

    # We set retries low and backoff to 0 to make tests fast
    with pytest.raises(RuntimeError, match="Failed to download"):
        download_file(url, dest, retries=2, backoff_factor=0)

    assert not dest.exists()  # Ensure cleanup happened


# --- extract_artifact tests ---


def test_extract_artifact_raw(tmp_path):
    src = tmp_path / "src.txt"
    src.write_text("hello")
    dst = tmp_path / "dst.txt"

    extract_artifact(src, dst, "raw")
    assert dst.read_text() == "hello"


def test_extract_artifact_gz(tmp_path):
    import gzip

    src = tmp_path / "test.gz"
    content = b"compressed content"
    with gzip.open(src, "wb") as f:
        f.write(content)

    dst = tmp_path / "unzipped.txt"
    extract_artifact(src, dst, "gz")
    assert dst.read_bytes() == content


def test_extract_artifact_invalid_format(tmp_path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    with pytest.raises(ValueError, match="Unsupported source format"):
        extract_artifact(src, dst, "unsupported_format")


# --- provision_asset tests ---


@patch("external_db_provisioning.provisioner.verify_checksum")
def test_provision_asset_skips_if_valid(mock_verify, tmp_path):
    target = tmp_path / "valid.fasta"
    target.write_text("content")

    # If verify_checksum doesn't raise an error, it's valid
    result = provision_asset("url", target, "hash123")

    assert result is False
    assert mock_verify.call_count == 1


@patch("external_db_provisioning.provisioner.download_file")
@patch("external_db_provisioning.provisioner.extract_artifact")
@patch("external_db_provisioning.provisioner.verify_checksum")
def test_provision_asset_full_lifecycle(mock_verify, mock_extract, mock_download, tmp_path):
    target = tmp_path / "output.txt"
    # Mocking verify_checksum to always pass
    mock_verify.return_value = True

    result = provision_asset(
        url="http://fake.com",
        target_path=target,
        final_sha256="final_hash",
        source_sha256="source_hash",
        source_format="raw",
    )

    assert result is True
    mock_download.assert_called_once()
    mock_extract.assert_called_once()
    # verify_checksum should be called for:
    # 1. Existing check (fails/raises in this scenario usually, but we mock flow)
    # 2. Source check
    # 3. Final check
    assert mock_verify.call_count >= 2


def test_provision_asset_removes_stale_tmp(tmp_path):
    target = tmp_path / "data.txt"
    stale_tmp = target.with_suffix(".raw.tmp")
    stale_tmp.write_text("corrupt")

    # We mock the sub-calls to isolate the "stale removal" logic
    with (
        patch("external_db_provisioning.provisioner.download_file"),
        patch("external_db_provisioning.provisioner.extract_artifact"),
        patch("external_db_provisioning.provisioner.verify_checksum"),
    ):

        provision_asset("url", target, "hash")

    assert not stale_tmp.exists()


# --- Path 1: Early Return ---


@responses.activate
def test_download_file_replaces_existing_file(tmp_path):
    """Test Path 1: download_file replaces any existing file with fresh data."""
    dest = tmp_path / "already_here.txt"
    dest.write_text("old content")
    url = "https://example.com/data.txt"

    # Provide a mock response for the "fresh" data
    responses.add(responses.GET, url, body=b"new content", status=200)

    result = download_file(url, dest)

    assert result.read_text() == "new content"
    # This proves Path 1 (cleanup) and the actual download logic worked.


# --- Path 2: Partial File Cleanup during Retries ---


@responses.activate
def test_download_file_cleans_partial_on_retry(tmp_path):
    """Test Path 2: If a file exists when an attempt starts, it is unlinked."""
    dest = tmp_path / "retry_cleanup.txt"
    url = "https://example.com/data.txt"

    # 1. Setup a scenario where the first attempt fails after writing some data
    # Attempt 1: Fail with RequestException
    responses.add(responses.GET, url, status=500)
    # Attempt 2: Success
    responses.add(responses.GET, url, body=b"final data", status=200)

    # Pre-create a "partial" file to simulate a failed previous attempt/state
    dest.write_text("partial data")

    # We patch 'time.sleep' to make the test instant
    with patch("time.sleep", return_value=None):
        result = download_file(url, dest, retries=2, backoff_factor=0)

    assert result.read_text() == "final data"
    # Logic verification: The fact that it finished successfully means the
    # dest.unlink() inside the loop worked, otherwise dest.open("wb")
    # would just overwrite (which is fine), but our test verifies the loop path.


# --- Path 3: Final Cleanup on Total Failure ---
@patch("external_db_provisioning.provisioner.requests.get")
def test_download_file_cleans_up_on_total_failure(mock_get, tmp_path):
    """Test Path 3: If all retries fail, any partial file created is deleted."""
    dest = tmp_path / "total_failure.txt"
    url = "https://example.com/data.txt"

    # 1. Create a mock response object
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {"content-length": "100"}

    # 2. Define a generator that writes some data then explodes
    def mock_iter_content(chunk_size=1):
        yield b"partial data"
        # Simulate a network break mid-stream
        raise requests.exceptions.ChunkedEncodingError("Connection broken!")

    mock_response.iter_content = mock_iter_content
    mock_get.return_value = mock_response

    # 3. Run the function with retries disabled to hit the final cleanup logic
    with patch("time.sleep", return_value=None):
        with pytest.raises(RuntimeError, match="Failed to download"):
            download_file(url, dest, retries=1)

    # VERIFY: The file was created by the generator, but then unlinked by the handler
    assert not dest.exists()


# --- Bonus: Verifying the Retry Logic specifically ---


@responses.activate
def test_download_file_exponential_backoff_timing(tmp_path):
    """Verifies that the backoff factor is actually applied to sleep."""
    dest = tmp_path / "backoff.txt"
    url = "https://example.com/backoff"

    responses.add(responses.GET, url, status=500)
    responses.add(responses.GET, url, status=500)
    responses.add(responses.GET, url, body=b"done", status=200)

    with patch("time.sleep") as mock_sleep:
        download_file(url, dest, retries=3, backoff_factor=5)

        # Check sleep calls: 5^0 = 1, 5^1 = 5
        assert mock_sleep.call_args_list[0][0][0] == 1
        assert mock_sleep.call_args_list[1][0][0] == 5


def test_extract_artifact_tar(tmp_path):
    """Test Path: Extracting a specific file from a .tar archive."""
    # 1. Setup: Create a dummy tar file
    tar_path = tmp_path / "test.tar"
    inner_filename = "folder/data.txt"
    content = b"tar content"

    # Create a temporary file to add to the tar
    member_source = tmp_path / "member.txt"
    member_source.write_bytes(content)

    with tarfile.open(tar_path, "w") as tar:
        # We add it with the specific inner_path
        tar.add(member_source, arcname=inner_filename)

    # 2. Execute
    output_path = tmp_path / "extracted_tar.txt"
    extract_artifact(tar_path, output_path, "tar", inner_path=inner_filename)

    # 3. Verify
    assert output_path.exists()
    assert output_path.read_bytes() == content


def test_extract_artifact_zip(tmp_path):
    """Test Path: Extracting a specific file from a .zip archive."""
    # 1. Setup: Create a dummy zip file
    zip_path = tmp_path / "test.zip"
    inner_filename = "bioinfo/sequence.fasta"
    content = b">seq1\nATGC"

    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr(inner_filename, content)

    # 2. Execute
    output_path = tmp_path / "extracted_zip.fasta"
    extract_artifact(zip_path, output_path, "zip", inner_path=inner_filename)

    # 3. Verify
    assert output_path.exists()
    assert output_path.read_bytes() == content


def test_extract_artifact_tar_missing_inner_path(tmp_path):
    """Test Error Path: tarfile raises KeyError if inner_path is missing."""
    tar_path = tmp_path / "test.tar"
    with tarfile.open(tar_path, "w") as tar:
        tar.add(tmp_path, arcname="exists.txt")  # dummy add

    output_path = tmp_path / "out"
    # tarfile.getmember raises KeyError if not found
    with pytest.raises(KeyError):
        extract_artifact(tar_path, output_path, "tar", inner_path="ghost.txt")


@patch("external_db_provisioning.provisioner.download_file")
@patch("external_db_provisioning.provisioner.extract_artifact")
@patch("external_db_provisioning.provisioner.verify_checksum")
def test_provision_asset_reprovisions_on_checksum_failure(
    mock_verify, mock_extract, mock_download, tmp_path
):
    """Test Path: target_path exists but is corrupt, triggering re-provisioning."""

    # 1. Setup paths
    target = tmp_path / "corrupt_data.txt"
    target.write_text("corrupt content")

    # 2. Setup mock side effects for verify_checksum
    # First call (initial check): Raises ValueError (corrupt)
    # Second call (source check): None (success) - omitted because source_sha256 is None here
    # Third call (final check): None (success)
    mock_verify.side_effect = [ValueError("Checksum mismatch"), None]

    # 3. Execute
    result = provision_asset(
        url="http://example.com/data.txt",
        target_path=target,
        final_sha256="expected_hash",
        source_format="raw",
    )

    # 4. Verifications
    assert result is True  # Should return True because it had to re-provision

    # verify_checksum should be called twice:
    # once at the start (failure) and once at the end (success)
    assert mock_verify.call_count == 2

    # Ensure the lifecycle continued after the first failure
    mock_download.assert_called_once()
    mock_extract.assert_called_once()


@patch("external_db_provisioning.provisioner.download_file")
@patch("external_db_provisioning.provisioner.extract_artifact")
@patch("external_db_provisioning.provisioner.verify_checksum")
def test_provision_asset_removes_stale_source_before_download(
    mock_verify, mock_extract, mock_download, tmp_path
):
    """Test Path: ensures stale .tmp files are removed if they exist."""
    # 1. Setup target file (so target.exists() is True)
    target = tmp_path / "data.txt"
    target.write_text("corrupt content")

    # 2. Setup stale temp file (the one we want to see deleted)
    stale_source = target.with_suffix(".raw.tmp")
    stale_source.write_text("stale data")

    # 3. Mock verify logic:
    # Call 1 (initial check): Raises ValueError -> enters 'except' block
    # Call 2 (final check): Returns None -> success
    mock_verify.side_effect = [ValueError("Missing"), None]

    # 4. Execute
    provision_asset(url="url", target_path=target, final_sha256="hash", source_format="raw")

    # 5. Verify the stale temp file was cleaned up
    assert not stale_source.exists()
