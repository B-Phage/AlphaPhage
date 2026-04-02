#!/usr/bin/env python3
# USe with
# `export KMP_DUPLICATE_LIB_OK=TRUE && conda run -n training python tests/utils/inspection/inspect_checkpoint.py`

import argparse
import sys
from pathlib import Path

import torch


def inspect_checkpoint(checkpoint_path: str, verbose: bool = False):
    path = Path(checkpoint_path)
    if not path.exists():
        print(f"❌ Error: File not found at {checkpoint_path}")
        sys.exit(1)

    print(f"🔍 Inspecting: {path.name}")
    print("-" * 50)

    try:
        # Load metadata only (weights_only=False to see hparams)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        sys.exit(1)

    # 1. Basic Metadata
    print(f"{'Epoch:':<20} {checkpoint.get('epoch', 'N/A')}")
    print(f"{'Global Step:':<20} {checkpoint.get('global_step', 'N/A')}")
    print(f"{'PyTorch Version:':<20} {checkpoint.get('pytorch-lightning_version', 'N/A')}")

    # 2. Hyperparameters (The critical part for your refactor)
    hparams = checkpoint.get("hyper_parameters", {})

    if not hparams:
        print("\n⚠️  WARNING: No 'hyper_parameters' found. Model cannot be auto-reconstructed.")
    else:
        print("\n--- Model Configuration (hparams) ---")

        # Highlight the specific ones you're worried about
        critical_keys = ["num_tokens", "length_padded", "vocab_size_out"]

        for k, v in hparams.items():
            prefix = "⭐ " if k in critical_keys else "  - "
            # Handle nested DictConfigs/dicts
            if isinstance(v, dict):
                print(f"{prefix}{k}: (dict with {len(v)} keys)")
                if verbose:
                    for sub_k, sub_v in v.items():
                        print(f"      {sub_k}: {sub_v}")
            else:
                print(f"{prefix}{k:<18}: {v}")

    # 3. State Dict Summary (Layers)
    if verbose:
        print("\n--- Model Weights (state_dict) ---")
        state_dict = checkpoint.get("state_dict", {})
        print(f"Total Parameter Tensors: {len(state_dict)}")
        for i, (key, value) in enumerate(state_dict.items()):
            if i < 10:  # Only print first 10 unless really verbose
                print(f"  {key:<50} {tuple(value.shape)}")
        if len(state_dict) > 10:
            print(f"  ... and {len(state_dict) - 10} more.")

    print("-" * 50)


def main():
    # parser = argparse.ArgumentParser(description="Inspect PyTorch Lightning Checkpoint Metadata")
    # parser.add_argument("ckpt", type=str, help="Path to the .ckpt file")
    # parser.add_argument("-v", "--verbose", action="store_true", help="Print full dictionary contents and layer names")

    # args = parser.parse_args()
    # inspect_checkpoint(args.ckpt, args.verbose)

    snapshot_path = (
        "/Users/tamsen/Code/databases/inference/default_run_ep61_20260319/model/model_weights.ckpt"
    )
    inspect_checkpoint(snapshot_path, verbose=True)


if __name__ == "__main__":
    main()
