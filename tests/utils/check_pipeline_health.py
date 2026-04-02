# export KMP_DUPLICATE_LIB_OK=TRUE && conda run -n training python tests/utils/check_pipeline_health.py


import pathlib
import sys

import h5py
import pandas as pd


def check_snapshot_integrity(snapshot_path: str):
    root = pathlib.Path(snapshot_path)
    print(f"🔍 Checking Snapshot: {root.name}")

    # 1. Path Resolution
    phage_ref = root / "reference/vector_info/phage_vec_1_info.tsv"
    host_ref = root / "reference/vector_info/host_vec_1_info.tsv"

    errors = 0

    # 2. Test: Reference TSV Structure (The 'NROWS' Test)
    print("\n--- Test 1: Reference TSV Structure ---")
    for ref in [phage_ref, host_ref]:
        if not ref.exists():
            print(f"❌ Missing: {ref}")
            errors += 1
            continue

        # Try loading with and without nrows to see if it's 'Stacked'
        full_df = pd.read_csv(ref, sep="\t", header=None)
        row1_df = pd.read_csv(ref, sep="\t", header=None, nrows=1)

        if full_df.shape[0] > 1:
            print(f"⚠️  Warning: {ref.name} has {full_df.shape[0]} rows (Metadata detected).")
            print(f"✅ Safe Load Check: Header row contains {row1_df.shape[1]} columns.")
        else:
            print(f"✅ {ref.name} is a clean single-row header.")

    # 3. Test: The 5385 Dimension Match
    print("\n--- Test 2: Dimension Alignment ---")
    p_count = pd.read_csv(phage_ref, sep="\t", header=None, nrows=1).shape[1]
    h_count = pd.read_csv(host_ref, sep="\t", header=None, nrows=1).shape[1]
    total = p_count + h_count

    expected = 5385  # Your model's truth
    if total == expected:
        print(f"✅ Dimensions Match: {p_count}p + {h_count}h = {total} (Target: {expected})")
    else:
        print(f"❌ CRITICAL: Total width {total} != {expected}. Model will crash.")
        errors += 1

    # 4. Test: Vocabulary Token Leakage
    print("\n--- Test 3: Data Contract (DuckDB Pivot Safety) ---")
    # We check a sample variant shard to ensure no 'UNKNOWN_NEW_VARIANT' strings exist
    gallery_sample = list((root / "reference/phage/variant_shards").rglob("*.parquet"))
    if gallery_sample:
        df_sample = pd.read_parquet(gallery_sample[0])
        # Ensure variant_id is numeric or a short token, not a long string
        sample_val = str(df_sample["variant_id"].iloc[0])
        if len(sample_val) > 10 or not any(char.isdigit() for char in sample_val):
            print(f"❌ CRITICAL: Found 'Dirty' variant_id: '{sample_val}'. Pivot will explode.")
            errors += 1
        else:
            print(f"✅ Variant IDs look like clean tokens (e.g., '{sample_val}')")

    print("\n--- Result ---")
    if errors == 0:
        print("🚀 PASSED: Snapshot is healthy. Ready for inference.")
    else:
        print(f"🛑 FAILED: {errors} critical errors found. Fix these before running 'make'.")
        sys.exit(1)


if __name__ == "__main__":
    # Point this to your latest snapshot
    path = "/Users/tamsen/Code/databases/inference/default_run_ep3_20260324"
    check_snapshot_integrity(path)
