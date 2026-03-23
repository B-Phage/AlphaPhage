import sys
from pathlib import Path

# Add the packages directory to sys.path manually
# This finds the 'packages' folder relative to the script location
repo_root = Path(__file__).resolve().parents[2]  # Adjust the '2' based on depth
packages_path = repo_root / "packages"
if str(packages_path) not in sys.path:
    sys.path.append(str(packages_path))

from pathlib import Path

import h5py
import numpy as np
import pandas as pd


def convert_h5_to_tsv(h5_path: str, output_path: str | None = None):
    p = Path(h5_path)
    out = output_path or p.with_suffix(".tsv")

    try:
        # Most R&D code saves via df.to_hdf()
        df = pd.read_hdf(h5_path)
        df.to_csv(out, sep="\t")
        print(f"✅ Converted {h5_path} to {out}")
        print(f"📊 Shape: {df.shape}")
    except ValueError:
        # If it's not a Pandas format, it might be a raw NumPy array
        print(f"⚠️ {h5_path} isn't a standard Pandas HDF5. Attempting raw inspection...")
        with h5py.File(h5_path, "r") as f:
            for key in f.keys():
                print(f"Found key: {key}, Shape: {f[key].shape}, Dtype: {f[key].dtype}")
                # You could further extract f[key][:] to a CSV here if needed


def convert_h5_to_folder(h5_path: str):
    h5_file = Path(h5_path)
    if not h5_file.exists():
        print(f"❌ File not found: {h5_path}")
        return

    # Create a directory named after the h5 file (without extension)
    output_dir = h5_file.parent / h5_file.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"📂 Created directory: {output_dir}")

    try:
        # Strategy A: Attempt Pandas/PyTables Load
        df = pd.read_hdf(h5_path)
        df.to_csv(output_dir / "pandas_data.tsv", sep="\t")
        print(f"  ✅ Saved Pandas Table to 'pandas_data.tsv'")

    except Exception:
        # Strategy B: Manual Dataset Extraction (Your colleague's format)
        print(f"  ⚠️ Not a standard Pandas HDF5. Extracting raw datasets...")

        with h5py.File(h5_path, "r") as f:
            for key in f.keys():
                data = f[key][:]
                output_file = output_dir / f"{key}.tsv"

                # Handle different data types
                if data.dtype.kind in {"S", "U"}:  # Byte strings (|S50) or Unicode
                    # Convert byte strings back to readable text
                    clean_data = np.vectorize(
                        lambda x: x.decode("utf-8") if isinstance(x, bytes) else x
                    )(data)
                    df = pd.DataFrame(clean_data)
                else:
                    df = pd.DataFrame(data)

                df.to_csv(output_file, sep="\t", index=False, header=False)
                print(f"  ✅ Saved dataset '{key}' ({data.shape}) to {output_file.name}")


if __name__ == "__main__":
    # train_features_path = "tests/data/x_Test_50_50_TOK_test_AT_cond2_aligned_test_binary_pairkey.h5"
    # train_labels_path = "tests/data/cl_Test_50_50_TOK_test_AT_cond2_aligned_test_binary_pairkey.h5"

    # convert_h5_to_folder(train_features_path)
    # convert_h5_to_folder(train_labels_path)

    x_train_data = (
        "/Users/tamsen/Code/AlphaPhage/ml_assets/data/processed/model_training/x_train_data_v1.h5"
    )
    cl_train_data = (
        "/Users/tamsen/Code/AlphaPhage/ml_assets/data/processed/model_training/cl_train_data_v1.h5"
    )

    convert_h5_to_folder(x_train_data)
    convert_h5_to_folder(cl_train_data)

    # x_inf_data = (
    #     "/Users/tamsen/Code/databases/inference/output/x_inference_set_v1.h5"
    # )
    # convert_h5_to_folder(x_inf_data)
