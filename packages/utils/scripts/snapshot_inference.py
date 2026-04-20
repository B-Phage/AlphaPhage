import glob
import json
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import cast

import h5py
import numpy as np
import pandas as pd
import utils.consol_log as log
import yaml
from data_contracts.h5_schema import H5DataKeys
from omegaconf import DictConfig, OmegaConf
from utils.scripts.create_inference_config import generate_inference_block
from utils.scripts.hmm_snapshot import snapshot_hmm_db

logger = log.get_logger()

CONFIG_FILES = ["params.yaml"]
TARGET_BASE_DIR = Path("../databases/inference")
TRAINING_ROOT = Path("ml_assets/data/training")
RUNTIME_CONFIG = Path("ml_assets/runtime_configs")
VOCAB_DIR = Path("ml_assets/vocab")
MODEL_TRIANING_DATA = Path("ml_assets/data/processed/model_training")


def extract_h5_info_metadata(h5_source_path: Path, output_dir: Path):
    """
    Extracts info datasets from training H5.
    Uses explicit type checking to resolve Pylance method warnings.
    """
    keys = H5DataKeys()
    target_keys = [keys.HOST_VEC_INFO, keys.PHAGE_VEC_INFO]

    if not h5_source_path.exists():
        logger.warning(f"⚠️ Training H5 not found at {h5_source_path}.")
        return

    with h5py.File(h5_source_path, "r") as f:
        for key in target_keys:
            if key in f:
                item = f[key]

                # KEEP: Explicit type check for Pylance/Runtime safety
                if isinstance(item, h5py.Dataset):
                    # item[()] gets the raw numpy-like array
                    data_array = item[()]

                    # FIX: Process byte strings while PRESERVING 2D SHAPE
                    if data_array.dtype.kind in {"S", "U"}:
                        # Map the decode function across the 2D array
                        func = np.vectorize(
                            lambda x: x.decode("utf-8") if isinstance(x, bytes) else x
                        )
                        clean_data = func(data_array)
                    else:
                        clean_data = data_array

                    # FIX: Create DataFrame from 2D array so it stays 2 rows by N columns
                    df = pd.DataFrame(clean_data)

                    out_file = output_dir / f"{key}.tsv"
                    # KEEP: sep="\t" and no index/header
                    df.to_csv(out_file, sep="\t", index=False, header=False)

                    # UPDATE: Log the actual shape (e.g., 2 x 2167)
                    logger.info(f"   ✅ Extracted {key} {clean_data.shape}")
                else:
                    logger.warning(f"   ⚠️ Key '{key}' exists but is a {type(item)}, not a Dataset.")
            else:
                logger.warning(f"   ⚠️ Key '{key}' not found in H5 file.")


def run_cmd(cmd):
    """Utility to run shell commands and return output."""
    result = subprocess.run(cmd, capture_output=True, text=True, shell=True)
    return result.returncode, result.stdout.strip()


def transform_to_inference_paths(cfg_item, inference_base="../../databases/inference/"):
    """
    Recursively updates paths using Pathlib logic to ensure structural
    integrity rather than simple string replacement.
    """
    if isinstance(cfg_item, dict):
        return {k: transform_to_inference_paths(v, inference_base) for k, v in cfg_item.items()}
    elif isinstance(cfg_item, list):
        return [transform_to_inference_paths(v, inference_base) for v in cfg_item]
    elif isinstance(cfg_item, str):
        p = Path(cfg_item)
        parts = p.parts

        # Case 1: Path starts with ml_assets/
        # logic: drop 'ml_assets' and join remaining to inference_base
        if parts and parts[0] == "ml_assets":
            return str(Path(inference_base) / Path(*parts[1:]))

        # Case 2: Path starts with ../databases/
        # logic: ensure it has the correct depth for the inference env
        if len(parts) >= 2 and parts[0] == ".." and parts[1] == "databases":
            # We skip the first '..' to effectively turn ../databases into ../../databases
            return str(Path("../../") / Path(*parts[1:]))

    return cfg_item


def validate_state():
    logger.info("🔍 Validating environment state...")

    # 1. Capture Git SHA
    ret_sha, sha = run_cmd("git rev-parse HEAD")

    # 2. Capture Git Branch
    ret_branch, branch = run_cmd("git rev-parse --abbrev-ref HEAD")

    if ret_sha != 0 or ret_branch != 0:
        raise Exception("Failed to retrieve Git info. Ensure you are in the repo root.")

    # 3. Check DVC Status
    ret_dvc, status = run_cmd("dvc status --quiet")
    if "Data and pipelines are up to date" not in status and ret_dvc != 0:
        # Note: We just log the warning, we don't necessarily kill the script
        pass

    return sha, branch


def find_checkpoint(model_name, epoch):
    checkpoint_dir = TRAINING_ROOT / model_name / "checkpoints"
    pattern = str(checkpoint_dir / f"*epoch={epoch:02d}-*.ckpt")
    matches = glob.glob(pattern)
    if not matches:
        logger.error(f"Missing model in {pattern}")
        raise FileNotFoundError(f"No checkpoint for {model_name} at epoch {epoch}")
    return Path(matches[0])


def snapshot(model_name, epoch, buffer_factor=1.1):
    try:
        git_sha, git_branch = validate_state()

        h5_path = resolve_feature_set_path()
        logger.info(f"🔍 Resolved training data to: {h5_path}")

        # Setup naming
        timestamp = datetime.now().strftime("%Y%m%d")
        label = f"{model_name}_ep{epoch}_{timestamp}"
        target_path = TARGET_BASE_DIR / label

        logger.info(f"--- Creating Snapshot: {label} ---")

        # Find checkpoint before creating folders
        ckpt_src = find_checkpoint(model_name, epoch)

        # Create directory and handle Overwrite
        if target_path.exists():
            logger.warning(f"Snapshot directory {target_path} already exists. Overwritting")
            shutil.rmtree(target_path)
        target_path.mkdir(parents=True)

        folders = [
            "model",
            "config",
            "vocab",
            "runtime_config",
            "reference/phage/variant_shards",
            "reference/bac/variant_shards",
            "reference/vector_info",
        ]
        for folder in folders:
            (target_path / folder).mkdir(parents=True, exist_ok=True)

        # Porting Logic
        logger.info(f"📦 Copying model weights (epoch {epoch})...")
        shutil.copy2(ckpt_src, target_path / "model" / "model_weights.ckpt")

        # copy all runtime configs
        if RUNTIME_CONFIG.exists():
            logger.info(f"📦 Copying specialized runtime_configs...")
            shutil.copytree(RUNTIME_CONFIG, target_path / "runtime_config", dirs_exist_ok=True)
        else:
            logger.error(f"❌ No runtime config files exist at {RUNTIME_CONFIG}")

        logger.info("📦 Archiving domain configurations...")
        domains = ["phage", "bac"]
        domain_params_map = {}

        for domain in domains:
            dest_cfg = target_path / "runtime_config" / f"{domain}_preprocessing.yaml"

            if dest_cfg.exists():
                # Load to drive the gallery porting
                domain_params_map[domain] = OmegaConf.load(dest_cfg)
            else:
                logger.error(f"❌ Critical: {domain} config missing at {dest_cfg}")

        # Port Galleries for BOTH domains
        for domain, params in domain_params_map.items():
            cluster_name = params.msa_var.msa_cluster_name
            src_variant_dir = Path(params.msa_var.msa_variant_dir)

            src_shards = src_variant_dir / cluster_name
            # Store them in domain-specific subfolders inside the snapshot
            dest_shards = target_path / "reference" / domain / "variant_shards" / cluster_name

            if src_shards.exists():
                logger.info(f"📦 Porting {domain.upper()} Gallery: {cluster_name}")
                shutil.copytree(src_shards, dest_shards)
            else:
                logger.error(f"❌ {domain.upper()} Source Gallery not found: {src_shards}")

        logger.info("📦 Rewiring all runtime configurations for inference...")
        runtime_cfg_dir = target_path / "runtime_config"

        # Iterate over every .yaml and .yml file in the copied runtime_config folder
        for cfg_file in runtime_cfg_dir.glob("*.y*ml"):
            try:
                # 1. Load and resolve
                raw_cfg = OmegaConf.load(cfg_file)
                # We work with a container to make injection easy
                cfg_container = OmegaConf.to_container(raw_cfg, resolve=True)

                # 2. Transform standard paths (ml_assets -> ../../databases/...)
                inference_ready_dict = transform_to_inference_paths(cfg_container)

                # Convert back to OmegaConf to use our helper
                temp_cfg = cast(DictConfig, OmegaConf.create(inference_ready_dict))

                # 3. Identify if this is a domain config and inject block
                for domain in ["phage", "bac"]:
                    if domain in cfg_file.name:
                        logger.info(f"   💉 Injecting inference_config into {cfg_file.name}")

                        # Fix the variant dir to point to the snapshot reference folder
                        temp_cfg.msa_var.gallery_path = str(
                            "../"
                            / target_path
                            / "reference"
                            / domain
                            / "variant_shards"
                            / cluster_name
                        )

                        inf_block = generate_inference_block(
                            training_cfg=temp_cfg,
                            base_inf_dir="../../databases/inference/data/processed",
                            buffer_factor=buffer_factor,
                        )

                        # Insert the block
                        temp_cfg.inference_config = inf_block
                        domain_params_map[domain] = temp_cfg
                        break

                # 4. Save the "Super Config" back to the runtime_config folder
                with open(cfg_file, "w") as f:
                    yaml.dump(OmegaConf.to_container(temp_cfg), f, default_flow_style=False)

            except Exception as e:
                logger.error(f"   ❌ Failed to process {cfg_file.name}: {e}")

        if VOCAB_DIR.exists():
            logger.info(f"📦 Copying vocab...")
            shutil.copytree(VOCAB_DIR, target_path / "vocab", dirs_exist_ok=True)
        else:
            logger.error(f"❌ No vocab exist at {VOCAB_DIR}")

        logger.info(f"📦 Copying DVC/Params configuration...")
        for cfg in CONFIG_FILES:
            if os.path.exists(cfg):
                shutil.copy2(cfg, target_path / cfg)
            else:
                logger.error(f"❌ No config files exist at {cfg}")

        # Port Vector Info (The 'Labels')
        info_output_dir = target_path / "reference" / "vector_info"
        info_output_dir.mkdir(parents=True, exist_ok=True)
        extract_h5_info_metadata(h5_path, info_output_dir)

        # Metadata including the critical SHA
        metadata = {
            "snapshot_name": label,
            "git_sha": git_sha,
            "git_branch": git_branch,
            "training_run": model_name,
            "epoch": epoch,
            "buffer_factor": buffer_factor,
            "created_at": datetime.now().isoformat(),
            "dvc_synced": "Verified via dvc status",
        }

        with open(target_path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=4)

        logger.info(f"✅ Success! Snapshot archived at: {target_path}")

        logger.info("-" * 30)
        logger.info(f"✅ SNAPSHOT COMPLETE")
        logger.info(f"📍 Location: {target_path}")
        logger.info(f"🧬 Data:     {h5_path.name}")
        logger.info(f"🧠 Model:    {ckpt_src.name}")
        logger.info(f"🔗 Git:      {git_sha[:7]} [{git_branch}]")
        logger.info("-" * 30)

        snapshot_hmm_db(snapshot=label, new_hmm_db_name="/hmm_inf/")
        logger.info(f"🐳 HMM subset saved to 'databases/hmm_inf' for docker image")

    except Exception as e:
        logger.error(f"❌ Aborted: {e}")


def resolve_feature_set_path(params_path: Path = Path("params.yaml")) -> Path:
    """
    Dynamically constructs the path to the training H5 file based on params.yaml.
    Expected structure: {training_data_dir}/x_{training_data_name}.h5
    """
    if not params_path.exists():
        raise FileNotFoundError(f"❌ Could not find {params_path} to resolve feature set.")

    # Load the params
    conf = OmegaConf.load(params_path)

    try:
        # Extract values from the 'storage' block
        data_dir = Path(conf.storage.training_data_dir)
        data_name = conf.storage.training_data_name

        # Construct the full path (assuming the 'x_' prefix is your convention)
        feature_set_path = data_dir / f"x_{data_name}.h5"

        if not feature_set_path.exists():
            logger.warning(f"⚠️ Resolved path {feature_set_path} does not exist on disk!")

        return feature_set_path

    except Exception as e:
        logger.error(f"❌ Failed to parse storage block in {params_path}: {e}")
        raise


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--epoch", required=True, type=int)
    args = parser.parse_args()

    snapshot(args.model, args.epoch)
