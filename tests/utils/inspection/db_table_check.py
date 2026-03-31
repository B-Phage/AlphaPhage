import sys
from pathlib import Path

# Add the packages directory to sys.path manually
# This finds the 'packages' folder relative to the script location
repo_root = Path(__file__).resolve().parents[2]  # Adjust the '2' based on depth
packages_path = repo_root / "packages"
if str(packages_path) not in sys.path:
    sys.path.append(str(packages_path))

from storage_io.duckdb_client import DuckDBClient

# Example Usage:
db_path = "ml_assets/data/processed/aggregated/features.db"
tbl_name = "bacteria_variants_v1"

with DuckDBClient(db_path) as client:
    # 1. Inspect the 'phage_variants' table
    try:
        df = client.head(tbl_name, n=100)

        # 2. Print to console for quick check
        print(df)

        # 3. Save to a CSV for deeper inspection
        df.to_csv("inspection_results.csv", index=False)

    except ValueError as e:
        print(f"Inspection failed: {e}")
