import gzip
import shutil
import tarfile
import time
import zipfile
from pathlib import Path
from typing import Optional

import requests
import utils.consol_log as log
from tqdm import tqdm
from utils.hash import verify_checksum

logger = log.get_logger()


def download_file(url: str, dest_path: Path, retries: int = 3, backoff_factor: int = 2) -> Path:
    """Downloads a file with a progress bar and exponential backoff retries.

    This function streams a file from a URL to a local destination path. It
    implements a robust download strategy that handles intermittent network
    failures by retrying with an increasing delay between attempts. To ensure
    data integrity, any partial or failed downloads are unlinked before a
    retry or final failure.

    Args:
        url (str): The direct URL of the file to download.
        dest_path (Path): The local filesystem path where the file should be saved.
        retries (int, optional): Total number of download attempts to make.
            Defaults to 3.
        backoff_factor (int, optional): The base for exponential backoff
            calculation (backoff_factor ** attempt). Defaults to 2.

    Returns:
        Path: The path to the successfully downloaded file.

    Raises:
        RuntimeError: If the download fails after all retry attempts are exhausted.
        requests.exceptions.HTTPError: If the server returns an unsuccessful
            status code.
        IOError: If there are issues creating directories or writing the file
            to disk.

    Note:
        The function uses 'tqdm' for progress visualization and ensures the
        temporary file is deleted if the process is interrupted, maintaining
        an atomic-like download state.
    """

    dest_path.parent.mkdir(parents=True, exist_ok=True)

    # We store the last exception to raise it if all retries fail
    last_exception: Optional[Exception] = None

    for attempt in range(retries):
        try:
            # If a previous attempt left a partial file, clear it before trying again
            if dest_path.exists():
                logger.info(f"Cleaning partial file before attempt {attempt + 1}")
                dest_path.unlink()

            logger.info(f"Downloading (Attempt {attempt + 1}/{retries}): {url}")

            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0))

            # Using 'with' for tqdm ensures it closes even if an error occurs mid-download
            with tqdm(
                total=total_size, unit="iB", unit_scale=True, desc=dest_path.name
            ) as progress_bar:
                with dest_path.open("wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            progress_bar.update(len(chunk))

            return dest_path

        except (requests.exceptions.RequestException, IOError) as e:
            last_exception = e
            logger.warning(f"Attempt {attempt + 1} failed: {e}")

            if attempt < retries - 1:
                sleep_time = backoff_factor**attempt
                logger.info(f"Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)

    # If we reach here, all retries failed
    if dest_path.exists():
        dest_path.unlink()

    raise RuntimeError(
        f"Failed to download {url} after {retries} attempts. Last error: {last_exception}"
    )


def extract_artifact(
    source_path: Path, artifact_path: Path, source_format: str, inner_path: str | None = None
) -> None:
    """Decompresses or copies a source file to a destination based on its format.

    This function handles the 'unpacking' phase of the provisioning lifecycle.
    It supports direct file copying for raw data as well as extraction for
    common bioinformatics archive formats. For multi-file archives (tar/zip),
    it allows for the extraction of a specific internal file.

    Args:
        source_path (Path): The path to the downloaded (potentially compressed)
            source file.
        artifact_path (Path): The final destination path for the extracted
            or copied file.
        source_format (str): The format of the source file. Supported values:
            'raw', 'gz', 'tar', 'zip'.
        inner_path (str | None, optional): The internal path/filename within
            a 'tar' or 'zip' archive to be extracted. Required if the
            format is 'tar' or 'zip'. Defaults to None.

    Raises:
        ValueError: If an unsupported 'source_format' is provided.
        KeyError: If 'inner_path' is required but cannot be found within
            the archive.
        IOError: If there are issues reading the source or writing to the
            destination.

    Example:
        >>> extract_artifact(
        ...     Path("data.tar.tmp"),
        ...     Path("data.txt"),
        ...     "tar",
        ...     inner_path="subdir/data.txt"
        ... )
    """
    if source_format == "raw":
        shutil.copyfile(source_path, artifact_path)
    elif source_format == "gz":
        with gzip.open(source_path, "rb") as src, artifact_path.open("wb") as dst:
            shutil.copyfileobj(src, dst)
    elif source_format == "tar":
        with tarfile.open(source_path, "r:*") as tar:
            member = tar.getmember(inner_path)  # type: ignore[arg-type]
            # Specify the union-attr code to satisfy strict Mypy rules
            with tar.extractfile(member) as src, artifact_path.open("wb") as dst:  # type: ignore[union-attr]
                shutil.copyfileobj(src, dst)

    elif source_format == "zip":
        with zipfile.ZipFile(source_path) as zf:
            # zipfile.open returns an object that always has __enter__
            with zf.open(inner_path) as src, artifact_path.open("wb") as dst:  # type: ignore[arg-type]
                dst.write(src.read())
    else:
        raise ValueError(f"Unsupported source format: {source_format}")


def provision_asset(
    url: str,
    target_path: Path,
    final_sha256: str,
    source_sha256: str | None = None,
    source_format: str = "raw",
    inner_path: str | None = None,
) -> bool:
    """Orchestrates the full lifecycle of an external data asset.

    This function implements a 'verify-first' provisioning pattern. It follows
     a strict sequence: Check existing -> Download -> Verify Source ->
    Extract -> Verify Final. It ensures idempotency by skipping the setup
    if a valid file already exists, and ensures integrity by re-provisioning
    if a checksum mismatch is detected.

    Args:
        url (str): The remote location of the asset or archive.
        target_path (Path): The final local destination for the processed asset.
        final_sha256 (str): The expected SHA-256 hash of the final, uncompressed
            asset on disk.
        source_sha256 (str | None, optional): The expected SHA-256 hash of the
            downloaded source (e.g., the compressed .tar.gz). If provided,
            integrity is checked before extraction. Defaults to None.
        source_format (str, optional): The format of the source (e.g., 'raw',
            'gz', 'tar', 'zip'). Defaults to 'raw'.
        inner_path (str | None, optional): For 'tar' or 'zip' formats, the
            path to the specific file inside the archive to extract.
            Defaults to None.

    Returns:
        bool: True if the asset was newly provisioned or updated. False if the
            existing asset was already valid and setup was skipped.

    Raises:
        ValueError: If a checksum verification fails at either the source
            or final stage.
        RuntimeError: If the download or extraction process fails.

    Note:
        This function is 'self-healing' and will automatically remove stale
        temporary files or invalid existing artifacts before attempting
        a re-provision.
    """
    if target_path.exists():
        try:
            verify_checksum(target_path, final_sha256)
            logger.info(f"Asset {target_path.name} is valid. Skipping setup.")
            return False
        except ValueError:
            logger.warning(f"Asset {target_path.name} failed checksum. Re-provisioning.")

    # Setup temp/source path
    source_path = target_path.with_suffix(f".{source_format}.tmp")

    if source_path.exists():
        logger.info(f"Removing stale temporary file: {source_path}")
        source_path.unlink()

    download_file(url, source_path)
    if source_sha256:
        verify_checksum(source_path, source_sha256)

    extract_artifact(source_path, target_path, source_format, inner_path)
    verify_checksum(target_path, final_sha256)

    # Cleanup source if it's not the final artifact
    if source_path != target_path:
        source_path.unlink(missing_ok=True)

    return True
