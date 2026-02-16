"""
HMM reference data setup workflow with YAML registry support.

Responsibilities:
- Fetch multi-HMM files from any source URL
- Handle different archive formats (tar, zip, raw)
- Verify checksums at source and final artifact stages
- Skip download/extraction if artifact already exists
- Split multi-HMM files into individual HMMs for downstream usage
- Provide a reproducible, idempotent setup step for pipelines consuming HMMs

Depends on:
- `packages/hmm/io.py` for HMM file splitting
- `gdown` for Google Drive downloads
- `PyYAML` for reading YAML registry
"""

import argparse
import gzip
import hashlib
import shutil
import tarfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Union

import utils.consol_log as log
import yaml  # type: ignore[import-untyped]
from hmm.io import split_hmms
from utils.hash import verify_checksum

logger = log.get_logger(level=log.INFO)


# -----------------------------
# Utility functions
# -----------------------------


def final_artifact_is_valid(final_path: Path, expected_sha256: str) -> bool:
    """Validates the existence and integrity of a final output artifact.

    Checks if the file at the specified path exists and performs a SHA-256
    checksum verification to ensure the file has not been corrupted or
    tampered with.

    Args:
        final_path (Path): The filesystem path to the artifact being checked.
        expected_sha256 (str): The hex-encoded SHA-256 hash expected for the file.

    Returns:
        bool: True if the file exists and the checksum matches the expected value.
            Returns False if the file is missing or if the checksum is incorrect.

    Note:
        This function internally catches `ValueError` from ``verify_checksum``
        to provide a boolean result instead of raising an exception.

    Example:
        >>> path = Path("output/model.hmm")
        >>> hash_val = "hash"
        >>> if final_artifact_is_valid(path, hash_val):
        ...     print("Artifact is ready for deployment.")
    """
    if not final_path.exists():
        return False
    try:
        verify_checksum(final_path, expected_sha256)
        return True
    except ValueError:
        return False


def download_file(url: str, dest_path: Path) -> Path:
    """Downloads a file from a URL to a local destination if it does not already exist.

    This function checks for the presence of a file at the target destination.
    If missing, it streams the content from the provided URL and writes it
    to the filesystem. It uses a buffered copy to remain memory-efficient
    for large files.

    Args:
        url (str): The web address (HTTP/HTTPS/FTP) of the file to download.
        dest_path (Path): A ``pathlib.Path`` object representing where the
            file should be saved.

    Returns:
        Path: The path to the downloaded (or already existing) file.

    Raises:
        urllib.error.URLError: If the connection fails or the URL is invalid.
        OSError: If there is an issue writing the file to the local disk.

    Example:
        >>> remote_url = "https://example.com/data.hmm"
        >>> local_path = Path("downloads/data.hmm")
        >>> download_file(remote_url, local_path)
        PosixPath('downloads/data.hmm')
    """
    if dest_path.exists():
        logger.info(f"{dest_path} already exists, skipping download.")
        return dest_path
    logger.info(f"Downloading from {url} to {dest_path}")
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as r, dest_path.open("wb") as f:
        shutil.copyfileobj(r, f)
    return dest_path


# -----------------------------
# Format-aware extraction
# -----------------------------
def extract_from_tar(archive_path: Path, member_path: str, dest_path: Path) -> None:
    with tarfile.open(archive_path, "r:*") as tar:
        member = tar.getmember(member_path)
        extracted_file = tar.extractfile(member)
        if extracted_file is None:
            raise ValueError(
                f"Cannot extract {member_path} from {archive_path}: not a regular file"
            )
        with extracted_file as src, dest_path.open("wb") as dst:
            shutil.copyfileobj(src, dst)


def extract_from_zip(archive_path: Path, member_path: str, dest_path: Path) -> None:
    with zipfile.ZipFile(archive_path) as zf:
        with zf.open(member_path) as src, dest_path.open("wb") as dst:
            dst.write(src.read())


def extract_from_gz(archive_path: Path, dest_path: Path) -> None:
    """Decompress a single .gz file to dest_path."""
    with gzip.open(archive_path, "rb") as src, dest_path.open("wb") as dst:
        shutil.copyfileobj(src, dst)
    logger.info(f"Extracted {archive_path} -> {dest_path}")


def materialize_hmm_artifact(
    source_path: Path,
    artifact_path: Path,
    source_format: str,
    inner_path: str | None = None,
) -> None:
    """Produce the canonical HMM artifact from an array of sources."""
    if source_format == "raw":
        shutil.copyfile(source_path, artifact_path)
    elif source_format == "gz":
        extract_from_gz(source_path, artifact_path)
    elif source_format == "tar":
        assert inner_path is not None, "inner_path missing in config"
        extract_from_tar(source_path, inner_path, artifact_path)
    elif source_format == "zip":
        assert inner_path is not None, "inner_path missing in config"
        extract_from_zip(source_path, inner_path, artifact_path)
    else:
        raise ValueError(f"Unsupported source format: {source_format}")

    logger.info(f"Artifact materialized at {artifact_path}")


# -----------------------------
# Core workflow
# -----------------------------
def fetch_and_split_hmm(
    hmm_name: str,
    source_url: str,
    dest_dir: Union[str, Path],
    final_sha256: str,
    source_sha256: str | None = None,
    source_format: str = "raw",
    inner_path: str | None = None,
    can_split: bool = False,
) -> Path:
    """
    Download, extract, verify, and split an HMM artifact.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    artifact_path = dest_dir / f"{hmm_name}"
    split_dir = dest_dir / "split"

    # Step 1: Skip if final artifact exists and valid
    if final_artifact_is_valid(artifact_path, final_sha256):
        logger.info(f"Artifact {artifact_path} already exists and is valid.")
    else:
        # Step 2: Download raw/archive if needed
        source_path = dest_dir / f"{hmm_name}_source"
        download_file(source_url, source_path)

        # Step 3: Verify source checksum if provided
        if source_sha256:
            verify_checksum(source_path, source_sha256)

        # Step 4: Extract artifact
        materialize_hmm_artifact(source_path, artifact_path, source_format, inner_path)

        # Step 5: Verify final artifact checksum
        verify_checksum(artifact_path, final_sha256)

    # Step 6: Split into individual HMMs
    if can_split:
        logger.info(f"Splitting {hmm_name} into individual models, saving to {split_dir}")
        split_dir.mkdir(exist_ok=True)
        split_hmms(str(artifact_path), str(split_dir))
        logger.info(f"HMM {hmm_name} splitting complete.")

    logger.info(f"HMM {hmm_name} setup complete.")

    return split_dir


# -----------------------------
# YAML registry
# -----------------------------
def fetch_from_registry(yaml_path: Path) -> None:
    with yaml_path.open("r") as f:
        datasets = yaml.safe_load(f)

    for entry in datasets:
        fetch_and_split_hmm(
            hmm_name=entry["name"],
            source_url=entry["source_url"],
            dest_dir=entry["dest_dir"],
            final_sha256=entry["final_sha256"],
            source_sha256=entry.get("source_sha256"),
            source_format=entry.get("source_format", "raw"),
            inner_path=entry.get("inner_path"),
            can_split=entry.get("can_split", False),
        )


# -----------------------------
# CLI
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="HMM reference data setup workflow")
    parser.add_argument(
        "--hmm-name", dest="hmm_name", help="Name of the HMM database (without extension)"
    )
    parser.add_argument("--source-url", dest="source_url", help="Google Drive download link")
    parser.add_argument("--dest-dir", dest="dest_dir", help="Destination directory to store HMMs")
    parser.add_argument("--final-sha256", dest="final_sha256", help="Expected SHA256 checksum")
    parser.add_argument("--source-sha256", dest="source_sha256", type=str, default=None)
    parser.add_argument(
        "--source-format",
        dest="source_format",
        type=str,
        choices=["raw", "tar", "zip"],
        default="raw",
    )
    parser.add_argument("--inner-path", dest="inner_path", type=str, default=None)
    parser.add_argument(
        "--yaml-registry",
        dest="yaml_registry",
        type=str,
        default=None,
        help="Optional YAML file listing multiple HMM datasets",
    )

    args = parser.parse_args()

    if args.yaml_registry:
        fetch_from_registry(Path(args.yaml_registry))
    elif args.hmm_name and args.source_url and args.dest_dir and args.final_sha256:
        fetch_and_split_hmm(
            hmm_name=args.hmm_name,
            source_url=args.source_url,
            dest_dir=args.dest_dir,
            final_sha256=args.final_sha256,
            source_sha256=args.source_sha256,
            source_format=args.source_format,
            inner_path=args.inner_path,
        )
    else:
        parser.error("Must provide either --yaml_registry or all single HMM arguments")


if __name__ == "__main__":
    main()
