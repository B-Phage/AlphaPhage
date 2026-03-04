import argparse
from pathlib import Path

import utils.consol_log as log
import yaml
from external_db_provisioning.provisioner import provision_asset

logger = log.get_logger()


def setup_assets(config_path: Path):
    """Parses a manifest file and provisions all defined external assets.

    This function acts as the entry point for database setup. It reads a YAML
    configuration (the manifest), standardizes the input into a list of
    asset definitions, and invokes the provisioner for each entry.

    Args:
        config_path (Path): The filesystem path to the YAML manifest file
            containing asset definitions.

    Raises:
        FileNotFoundError: If the config_path does not exist.
        KeyError: If a required key (e.g., 'source_url' or 'final_sha256')
            is missing from an entry in the manifest.
        RuntimeError: If the provisioning process (download or verification) fails.

    Example:
        >>> setup_assets(Path("ml_assets/config/kaptive_v0.0.1.setup.yaml"))
    """
    with config_path.open("r") as f:
        config = yaml.safe_load(f)

    # Standardize to a list so the logic is identical for 1 or N assets
    entries = config if isinstance(config, list) else [config]

    for entry in entries:
        name = entry.get("name")
        dest_dir = Path(entry.get("dest_dir", "."))
        target_path = dest_dir / name

        logger.info(f"Processing asset: {name}")

        provision_asset(
            url=entry["source_url"],
            target_path=target_path,
            final_sha256=entry["final_sha256"],
            source_sha256=entry.get("source_sha256"),
            source_format=entry.get("source_format", "raw"),
            inner_path=entry.get("inner_path"),
        )


def main():
    parser = argparse.ArgumentParser(description="Provision Kaptive external databases")
    parser.add_argument("--config", required=True, type=Path, help="Path to YAML config")
    args = parser.parse_args()

    if not args.config.exists():
        logger.error(f"Config file not found: {args.config}")
        return

    setup_assets(args.config)


if __name__ == "__main__":
    main()
