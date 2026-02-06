#!/usr/bin/env python3
"""
Compute SHA256 hashes for a list of core packages and write them
to separate output files for DVC.
"""

import argparse
import hashlib
from pathlib import Path


def hash_package(package_dir: Path) -> str:
    """Return SHA256 hash of all files in a package directory."""
    h = hashlib.sha256()
    for f in sorted(package_dir.rglob("*.py")):
        h.update(f.read_bytes())
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--packages", nargs="+", required=True, help="List of package directories to hash"
    )
    parser.add_argument("--output-dir", required=True, help="Directory to write hash files")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for pkg_path in args.packages:
        pkg = Path(pkg_path)
        if not pkg.exists():
            raise FileNotFoundError(f"Package directory not found: {pkg}")
        h = hash_package(pkg)
        hash_file = output_dir / f"{pkg.name}.hash"
        hash_file.write_text(h)
        print(f"Hashed {pkg} -> {hash_file} ({h[:8]}...)")


if __name__ == "__main__":
    main()
