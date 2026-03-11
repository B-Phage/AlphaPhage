import argparse
from pathlib import Path

import utils.consol_log as log
import yaml
from external_db_provisioning.provisioner import provision_asset
from hmm.io import split_hmms

logger = log.get_logger()


def setup_hmm(entry: dict):
    """Provisions a specific HMM asset and performs optional post-processing.

    This function coordinates the download and verification of an HMM file.
    If the HMM is successfully updated (or is missing its split components),
    it triggers a splitting process to break the multi-HMM file into
    individual model files for downstream processing.

    Args:
        entry (dict): A configuration dictionary representing a single asset.
            Expected keys include:
            - name (str): The filename for the downloaded asset.
            - dest_dir (str): Directory where the asset should be stored.
            - source_url (str): The URL to download the asset from.
            - final_sha256 (str): The expected SHA256 hash of the final file.
            - can_split (bool, optional): Whether to trigger HMM splitting.
                Defaults to False.
            - source_format (str, optional): Archive format (e.g., 'gz').
                Defaults to 'raw'.
            - inner_path (str, optional): Path within an archive to extract.

    Raises:
        KeyError: If mandatory keys are missing from the entry dictionary.
        RuntimeError: If the provisioning or splitting process fails.
        IOError: If directory creation or file writing is obstructed.
    """
    dest_dir = Path(entry["dest_dir"])
    dest_dir.mkdir(parents=True, exist_ok=True)

    artifact_path = dest_dir / entry["name"]

    # Use the generic provisioner
    was_updated = provision_asset(
        url=entry["source_url"],
        target_path=artifact_path,
        final_sha256=entry["final_sha256"],
        source_sha256=entry.get("source_sha256"),
        source_format=entry.get("source_format", "raw"),
        inner_path=entry.get("inner_path"),
    )

    # HMM-Specific Post-Processing
    if entry.get("can_split", False):
        split_dir = dest_dir / "split"
        # We only split if updated OR if the split dir is empty
        if was_updated or not any(split_dir.iterdir() if split_dir.exists() else []):
            logger.info(f"Splitting HMM: {entry['name']}")
            split_dir.mkdir(exist_ok=True)
            split_hmms(str(artifact_path), str(split_dir))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
        # Handle both single dict or list of dicts
        entries = config if isinstance(config, list) else [config]

    for entry in entries:
        setup_hmm(entry)


if __name__ == "__main__":
    main()
