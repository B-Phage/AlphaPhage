#!/usr/bin/env python3
"""
Compute SHA256 hashes for a list of core packages and write them
to separate output files for DVC.
"""

import argparse
from pathlib import Path

from utils.hash import hash_file, hash_package


def hash_path(path: Path) -> str:
    """
    Return SHA256 hash of a file or all files in a directory.
    - If path is a file, hashes just that file.
    - If path is a directory, recursively hashes all .py files.
    """
    if path.is_file():
        return hash_file(path)

    elif path.is_dir():
        return hash_package(path)

    else:
        raise FileNotFoundError(f"Path not found: {path}")


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
        path = Path(pkg_path)
        if not path.exists():
            raise FileNotFoundError(f"Path does not exist: {path}")
        h = hash_package(path)
        # Use path name if file, directory name if package
        name = path.stem if path.is_file() else path.name
        hash_file = output_dir / f"{name}.hash"
        hash_file.write_text(h)
        print(f"Hashed {path} -> {hash_file} ({h[:8]}...)")


if __name__ == "__main__":
    main()
