import argparse
import os
from pathlib import Path

import h5py
import pandas as pd
import torch
from data_contracts.h5_schema import H5DataKeys
from models.inference import TransformerInferencer
from omegaconf import OmegaConf


def main():
    parser = argparse.ArgumentParser(description="TransformerX Interaction Prediction CLI")
    parser.add_argument("--config", type=str, required=True, help="Path to params.yaml")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .ckpt file")
    parser.add_argument("--features", type=str, required=True, help="Path to input HDF5 features")
    parser.add_argument("--output", type=str, required=True, help="Path to save results (.csv)")
    args = parser.parse_args()

    # 1. Load context
    cfg = OmegaConf.load(args.config)
    keys = H5DataKeys()

    # 2. Initialize the "Gatekeeper" Inferencer
    # This automatically loads hparams from the checkpoint
    inferencer = TransformerInferencer(checkpoint_path=args.checkpoint, cfg=cfg)

    # 3. Run Prediction
    # We pass 'features' as 'labels' because the Dataset class just needs
    # the file handle to exist to initialize.
    print(f"🚀 Running inference on {args.features}...")
    probabilities = inferencer.predict(features_path=args.features, labels_path=args.features)

    # 4. Extract Pair Keys for mapping
    print("🔗 Mapping probabilities to interaction keys...")
    with h5py.File(args.features, "r") as f:
        # We assume pair_keys contains strings/bytes of HostID_PhageID
        raw_keys = f[keys.PAIR_KEYS][:]
        # Convert bytes to string if necessary
        pair_ids = [k.decode("utf-8") if isinstance(k, bytes) else str(k) for k in raw_keys]

    # 5. Build and Save Dataframe
    # Probabilities shape is usually [N, 2] for binary classification
    # Column 1 is typically the 'Positive' interaction (Class 1)
    df = pd.DataFrame(
        {
            "interaction_id": pair_ids,
            "probability_neg": probabilities[:, 0],
            "probability_pos": probabilities[:, 1],
        }
    )

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output, index=False)

    print(f"✨ Success! Predictions saved to: {args.output}")


if __name__ == "__main__":
    # Handle the OpenMP duplicate error for macOS hacks
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    main()
