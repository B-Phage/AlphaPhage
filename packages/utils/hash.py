# utils/hash.py
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from omegaconf import OmegaConf

DEFAULT_CHUNK_SIZE = 65536  # 64KB, consistent across functions


def hash_file(file_path: Path, chunk_size: int = DEFAULT_CHUNK_SIZE) -> str:
    """Compute SHA256 hash of a file."""
    h = hashlib.sha256()
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def file_manifest(
    files: Iterable[Path],
    relative_to: Path | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> dict[str, str]:
    """
    Build a deterministic manifest mapping file paths to SHA256 hashes.

    This function is the core primitive used for directory, package, and
    dataset hashing. Files are processed in a stable, sorted order to ensure
    reproducible results across runs. Each entry in the returned dictionary
    maps a file path to the SHA256 digest of its contents.

    Paths in the manifest can be stored relative to a provided root to
    ensure stability across machines and environments.

    Args:
        files: Iterable of file paths to include in the manifest. Non-file
            paths are ignored.
        relative_to: Optional base path used to store manifest keys as
            relative paths. If not provided, absolute paths are used.
        chunk_size: Size (in bytes) of chunks read when hashing files.
            Defaults to DEFAULT_CHUNK_SIZE.

    Returns:
        A dictionary mapping file path strings to SHA256 hex digests.

    Notes:
        - File paths are sorted before hashing to guarantee deterministic output.
        - Renaming or moving files will change the manifest, even if file
          contents remain the same.
        - This function does not write to disk; use `create_manifest` to
          persist the result as JSON.
    """
    manifest: dict[str, str] = {}

    for f in sorted(files, key=lambda p: str(p)):
        if not f.is_file():
            continue
        key = str(f.relative_to(relative_to)) if relative_to else str(f)
        manifest[key] = hash_file(f, chunk_size)

    return manifest


# relative_to: Path | None = None, #py3.10+ only
def manifest_from_directory(
    directory: Path, pattern: str = "*", relative_to: Optional[Path] = None
) -> dict[str, str]:
    files = directory.rglob(pattern)
    return file_manifest(files, relative_to=relative_to or directory.parent)


def hash_directory(directory: Path, pattern: str = "*") -> str:
    manifest = manifest_from_directory(directory, pattern)
    blob = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def hash_string(s: str) -> str:
    """Compute SHA256 hash of a string."""
    return hashlib.sha256(s.encode()).hexdigest()


def verify_checksum(file_path: Path, expected_sha256: str) -> None:
    """Verify a file matches the expected SHA256, raise ValueError if not."""
    actual = hash_file(file_path)
    if actual != expected_sha256:
        raise ValueError(
            f"Checksum mismatch for '{file_path}'. Expected {expected_sha256}, got {actual}."
        )


def hash_lake(lake_root: Path) -> str:
    """Hash all parquet files deterministically."""
    return hash_directory(lake_root, "*.parquet")


def create_manifest(
    files: Iterable[Path],
    output_path: Path,
    relative_to: Path | None = None,
):
    """
    Create a JSON manifest of file paths to SHA256 hashes.

    Args:
        files: Iterable of Path objects
        output_path: Path to write JSON manifest
        relative_to: If given, store paths relative to this
    """
    manifest = file_manifest(files, relative_to)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))


def hash_package(package_dir: Path) -> str:
    return hash_directory(package_dir, "*.py")


# config_items: Iterable[Any] | None = None, #py3.10+ only
def compute_combined_code_version(
    base_version: str,
    hash_files: list[Path],
    config_items: Optional[Iterable[Any]] = None,
) -> str:
    """
    Compute a deterministic code version identifier from code and configuration state.

    This function produces a single SHA256 digest that represents the combined
    identity of:
      1) A human-defined base version string (e.g., from a config file),
      2) One or more code hash files generated from core packages or scripts, and
      3) Optional configuration values or entire configuration sections.

    The resulting digest can be used as a stable `code_id` for orchestrators,
    caching systems, or workflow engines to determine when outputs must be
    recomputed due to changes in code or configuration.

    Args:
        base_version: A version string defined by developers (e.g., semantic
            version, git SHA, or config version) that represents the intended
            code release.
        hash_files: A list of paths to files containing precomputed SHA256
            hashes of core code dependencies. These files are read and their
            contents incorporated into the final digest.
        config_items: Optional iterable of configuration values, paths, or
            OmegaConf sections to include in the version calculation. These
            items are normalized into a deterministic, JSON-serializable form
            before hashing.

    Returns:
        A SHA256 hex digest string representing the combined code and
        configuration version.

    Notes:
        - Hash files are processed in sorted order to ensure determinism.
        - Configuration items are normalized to avoid ordering or type-related
          nondeterminism.
        - Any change in base version, code hashes, or included configuration
          will produce a new version identifier.
    """
    code_hashes = {str(p): p.read_text().strip() for p in sorted(hash_files, key=lambda p: str(p))}

    config_payload = {}
    if config_items:
        for i, item in enumerate(config_items):
            config_payload[f"cfg_{i}"] = _normalize(item)

    payload = {
        "base_version": base_version,
        "code_hashes": code_hashes,
        "config": config_payload,
    }

    return hash_string(json.dumps(payload, sort_keys=True))


def _normalize(value: Any) -> Any:
    """Convert values to JSON-serializable, deterministic form."""
    if isinstance(value, Path):
        return str(value.resolve())

    if OmegaConf.is_config(value):
        value = OmegaConf.to_container(value, resolve=True)

    if isinstance(value, dict):
        return {str(k): _normalize(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}

    if isinstance(value, (list, tuple, set)):
        return [_normalize(v) for v in sorted(value, key=lambda x: str(x))]

    return value
