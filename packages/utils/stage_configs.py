import argparse
import os
import shutil
from pathlib import Path
from typing import Any, Dict

import utils.consol_log as log
import yaml

logger = log.get_logger()


def stage_runtime_configs(params_path: str) -> None:
    """
    Parses the parameters file and copies source configuration files
    to their designated runtime locations.
    """
    if not os.path.exists(params_path):
        logger.error(f"Parameters file not found: {params_path}")
        raise FileNotFoundError(f"Parameters file not found at: {params_path}")

    with open(params_path, "r") as f:
        params: Dict[str, Any] = yaml.safe_load(f)

    # Counter for the summary
    staged_count = 0

    # Filter and process blocks containing 'source' and 'runtime' keys
    for key, config_block in params.items():
        if (
            isinstance(config_block, dict)
            and "source" in config_block
            and "runtime" in config_block
        ):

            src = config_block["source"]
            dst = config_block["runtime"]

            # Ensure the source file exists before attempting copy
            if not os.path.exists(src):
                logger.warning(f"Warning: Source file for '{key}' missing at {src}. Skipping.")
                continue

            # Load the source content
            with open(src, "r") as f:
                config_data = yaml.safe_load(f) or {}

            config_data["self_path"] = dst

            # INJECTION: Set self_path to the RUNTIME path
            # This ensures ${self_path} in dvc.yaml always resolves correctly
            dst_path = Path(dst)
            dst_path.parent.mkdir(parents=True, exist_ok=True)

            # Write the "augmented" config to the runtime location
            try:
                with open(dst_path, "w") as f:
                    yaml.dump(config_data, f, default_flow_style=False)
                logger.info(f"STAGING & INJECTING: {key} | {src} -> {dst}")
                staged_count += 1
            except Exception as e:
                logger.error(f"Failed to write to {dst}: {e}")
                raise

    logger.info(f"Successfully staged {staged_count} config files.")


def run_cli() -> None:
    """Entry point for CLI execution."""
    parser = argparse.ArgumentParser(description="DVC Config Staging Utility")
    parser.add_argument("--params", default="params.yaml")
    args = parser.parse_args()

    try:
        stage_runtime_configs(args.params)
    except Exception as e:
        logger.error(f"Staging failed: {e}")
        exit(1)


if __name__ == "__main__":
    run_cli()
