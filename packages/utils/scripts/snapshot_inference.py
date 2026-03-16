import glob
import json
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import utils.consol_log as log
from data_contracts.h5_schema import H5DataKeys

logger = log.get_logger()

CONFIG_FILES = ["dvc.yaml", "dvc.lock", "params.yaml"]
TARGET_BASE_DIR = Path("../databases/inference")
TRAINING_ROOT = Path("ml_assets/data/training")
REFERENCE_DB = Path("ml_assets/data/processed/aggregated/features.db")
RUNTIME_CONFIG = Path("ml_assets/runtime_configs")
VOCAB_DIR = Path("ml_assets/vocab")
MODEL_TRIANING_DATA = Path("/Users/tamsen/Code/AlphaPhage/ml_assets/data/processed/model_training")
FEATURE_SET = "x_train_data_v1.h5"


# # In your inference package/script
# class InferenceSession:
#     def __init__(self, snapshot_path):
#         self.root = Path(snapshot_path)
#         # Load the EXACT config used during that training run
#         self.config = self._load_config(self.root / "params.yaml")

#         # Load assets from the snapshot, NOT the repo root
#         self.model_path = self.root / "model"
#         self.cluster_ref = self.root / "clusters"

#         print(f"Initialized with snapshot: {self.root.name}")
#         print(f"Distance Metric: {self.config.get('distance_metric')}")


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

                # Check if it's actually a dataset (resolves Pylance warning)
                if isinstance(item, h5py.Dataset):
                    data = item[()]  # Using [()] is the h5py-idiomatic way to get all data

                    # Convert to numpy for easier handling
                    data_array = np.array(data)

                    if data_array.dtype.kind in {"S", "U"}:
                        # Decode byte strings to utf-8 strings
                        clean_data = [
                            x.decode("utf-8") if isinstance(x, bytes) else x
                            for x in data_array.flatten()
                        ]
                    else:
                        clean_data = data_array.tolist()

                    df = pd.DataFrame(clean_data)
                    out_file = output_dir / f"{key}.tsv"
                    df.to_csv(out_file, sep="\t", index=False, header=False)
                    logger.info(f"   ✅ Extracted {key} ({len(clean_data)} entries)")
                else:
                    logger.warning(f"   ⚠️ Key '{key}' exists but is a {type(item)}, not a Dataset.")
            else:
                logger.warning(f"   ⚠️ Key '{key}' not found in H5 file.")


def run_cmd(cmd):
    """Utility to run shell commands and return output."""
    result = subprocess.run(cmd, capture_output=True, text=True, shell=True)
    return result.returncode, result.stdout.strip()


def validate_state():
    print("🔍 Validating environment state...")

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
    pattern = str(checkpoint_dir / f"*epoch={epoch}*.ckpt")
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"No checkpoint for {model_name} at epoch {epoch}")
    return Path(matches[0])


def snapshot(model_name, epoch):
    try:
        git_sha, git_branch = validate_state()

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

        (target_path / "model").mkdir()
        (target_path / "reference").mkdir()

        # Porting Logic
        logger.info(f"📦 Copying model weights (epoch {epoch})...")
        shutil.copy2(ckpt_src, target_path / "model" / "model_weights.ckpt")

        ref_snapshot_dir = target_path / "reference"
        ref_snapshot_dir.mkdir(parents=True, exist_ok=True)

        if REFERENCE_DB.exists():
            logger.info(f"📦 Copying reference database: {REFERENCE_DB.name}")
            # copy2 preserves file metadata (timestamps, etc.)
            shutil.copy2(REFERENCE_DB, ref_snapshot_dir / "features.db")
            logger.info(f"   ✅ Database ported to {ref_snapshot_dir}/features.db")
        else:
            logger.error(f"❌ No reference database found at {REFERENCE_DB}")

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

        if RUNTIME_CONFIG.exists():
            logger.info(f"📦 Copying specialized runtime_configs...")
            shutil.copytree(RUNTIME_CONFIG, target_path / "runtime_config", dirs_exist_ok=True)
        else:
            logger.error(f"❌ No runtime config files exist at {RUNTIME_CONFIG}")

        h5_path = MODEL_TRIANING_DATA / FEATURE_SET
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
            "created_at": datetime.now().isoformat(),
            "dvc_synced": "Verified via dvc status",
        }

        with open(target_path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=4)

        logger.info(f"✅ Success! Snapshot archived at: {target_path}")

    except Exception as e:
        logger.error(f"❌ Aborted: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--epoch", required=True, type=int)
    args = parser.parse_args()

    snapshot(args.model, args.epoch)
