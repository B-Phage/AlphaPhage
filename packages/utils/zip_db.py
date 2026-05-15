import os
import zipfile
from typing import Any, Dict

import yaml


def load_config(config_path: str) -> Dict[str, Any]:
    """Parses the YAML configuration file.

    Args:
        config_path: The filesystem path to the YAML configuration file.

    Returns:
        A dictionary containing the configuration keys and values.
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def add_file_to_zip(zipf: zipfile.ZipFile, src: str, tgt: str) -> None:
    """Handles the actual writing of a single file to the archive.

    Args:
        zipf: An open zipfile.ZipFile object in write mode.
        src: The source path of the file on the local disk.
        tgt: The destination path/name of the file inside the ZIP archive.
    """
    print(f"  [File] {src} -> {tgt}")
    zipf.write(src, tgt)


def add_folder_to_zip(zipf: zipfile.ZipFile, src_root: str, tgt_root: str) -> None:
    """Walks through a directory and maps all files to the target ZIP path.

    Args:
        zipf: An open zipfile.ZipFile object in write mode.
        src_root: The root directory to start walking from.
        tgt_root: The prefix path to apply to all files inside the ZIP.
    """
    for root, _, files in os.walk(src_root):
        for file in files:
            full_src_path = os.path.join(root, file)
            # Calculate path relative to the source root
            rel_path = os.path.relpath(full_src_path, src_root)
            # Construct the internal ZIP path using the target prefix
            archive_path = os.path.join(tgt_root, rel_path)

            add_file_to_zip(zipf, full_src_path, archive_path)


def process_mapping(zipf: zipfile.ZipFile, src_raw: str, tgt_raw: str) -> None:
    """Validates the source and determines if it's a file or a folder.

    Args:
        zipf: An open zipfile.ZipFile object in write mode.
        src_raw: The raw source path from the config.
        tgt_raw: The raw target path from the config.
    """
    src = os.path.normpath(src_raw)
    tgt = os.path.normpath(tgt_raw)

    if not os.path.exists(src):
        print(f"Warning: Skipping {src} (Path not found)")
        return

    if os.path.isfile(src):
        add_file_to_zip(zipf, src, tgt)
    elif os.path.isdir(src):
        print(f"Processing Folder: {src}")
        add_folder_to_zip(zipf, src, tgt)


def create_mapped_zip(config_path: str) -> None:
    """Main orchestration function to create a ZIP based on a YAML config.

    This function reads the configuration, opens the target ZIP file,
    and iterates through all defined mappings to bundle files and folders.

    Args:
        config_path: The path to the YAML file containing 'target_zip'
            and 'mappings'.
    """
    config = load_config(config_path)
    target_zip = config.get("target_zip", "backup.zip")
    mappings = config.get("mappings", [])

    with zipfile.ZipFile(target_zip, "w", zipfile.ZIP_DEFLATED) as zipf:
        for entry in mappings:
            for src_path, tgt_path in entry.items():
                process_mapping(zipf, src_path, tgt_path)

    print(f"\nSuccessfully created {target_zip}")


if __name__ == "__main__":
    create_mapped_zip("config.yaml")
