import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import utils.consol_log as log

# Import our new components
from representation.feature_extractor import FeatureExtractor
from representation.h5_writer import H5TrainingWriter
from representation.metadata_manager import MetadataManager
from storage_io.duckdb_client import DuckDBClient

logger = log.get_logger()


def load_interaction_pairs(path: str) -> pd.DataFrame:
    """Helper to unpivot and filter unknown labels."""
    df = pd.read_csv(path, sep="\t", index_col=0)
    pairs = df.stack().reset_index()
    pairs.columns = ["host", "phage", "label"]
    return pairs[pairs["label"] != -1].copy()


def build_model_data(
    interaction_tsv: str,
    db_path: str,
    output_dir: str,
    base_filename: str,
    phage_table: str,
    host_table: str,
    serotype_table: str,
) -> None:
    """
    Orchestrates the construction of relational HDF5 training datasets.

    This function extracts unique host and phage feature vectors and maps them
    to interaction labels using integer indices. This "relational" approach
    avoids data redundancy by storing unique features once and using a
    pointer-based mapping (pair_indices) for each interaction.

    Args:
        interaction_tsv: Path to the tab-separated matrix containing
            ground-truth host-phage interactions.
        db_path: Path to the persistent DuckDB file containing feature tables.
        output_dir: Directory where the resulting HDF5 files will be saved.
        base_filename: Core name for output files (x_*.h5 and cl_*.h5).
        phage_table: Name of the DuckDB table containing phage variants.
        host_table: Name of the DuckDB table containing host variants.
        serotype_table: Name of the DuckDB table containing host serotypes.

    Returns:
        None: Saves 'x_[base_filename].h5' (features/indices) and
              'cl_[base_filename].h5' (labels) to output_dir.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Extraction (Unique Features Only)
    with DuckDBClient(db_path, read_only=True) as client:
        extractor = FeatureExtractor(client)
        # These dataframes now contain UNIQUE rows per organism
        phage_df = extractor.get_wide_phage_features(phage_table)
        host_df = extractor.get_wide_host_features(host_table, serotype_table)

        p_meta = extractor.get_metadata(phage_table)
        h_meta = extractor.get_metadata(host_table)

    # 2. Alignment & Index Mapping
    logger.info("🔗 Mapping interactions to integer indices...")
    interactions = load_interaction_pairs(interaction_tsv)

    # Create a mapping of Organism_ID -> Row_Index (0 to N-unique)
    phage_id_to_idx = {id_: i for i, id_ in enumerate(phage_df["organism_id"])}
    host_id_to_idx = {id_: i for i, id_ in enumerate(host_df["organism_id"])}

    # Filter interactions to only those where we have features for BOTH
    valid_mask = interactions["phage"].isin(phage_id_to_idx) & interactions["host"].isin(
        host_id_to_idx
    )
    data = interactions[valid_mask].copy()

    # Convert the string columns into integer pointers
    # These refer to the row number in the unique phage_df and host_df matrices
    phage_indices = data["phage"].map(phage_id_to_idx).values.astype("int32")
    host_indices = data["host"].map(host_id_to_idx).values.astype("int32")

    # This Nx2 array maps Interaction[i] -> [Host_Library[j], Phage_Library[k]]
    pair_indices = np.stack([host_indices, phage_indices], axis=1)

    # 3. Prepare Feature Matrices (No Redundancy)
    phage_cols = [c for c in phage_df.columns if c != "organism_id"]
    host_cols = ["k_type_dim1", "k_type_dim2"] + [
        c for c in host_df.columns if c not in ["organism_id", "k_type_dim1", "k_type_dim2"]
    ]

    # 4. Final Output (HDF5)
    writer = H5TrainingWriter(out_path, base_filename)
    payload = {
        # Unique libraries (Rows = Num Unique Organisms)
        "phage_vecs": phage_df[phage_cols].values.astype("int32"),
        "host_vecs": host_df[host_cols].values.astype("int32"),
        # Mapping and Labels (Rows = Num Total Interactions)
        "labels": data["label"].values.astype("int32"),
        "pair_indices": pair_indices,
        "pair_keys": data[["phage", "host"]].values.astype("S100"),
    }

    phage_info = MetadataManager.build_info_array(phage_cols, p_meta)
    host_info = MetadataManager.build_info_array(host_cols, h_meta)

    x_file, cl_file = writer.write(payload, phage_info, host_info)
    logger.info(f"✅ Created relational training set: {x_file.name} and {cl_file.name}")


def main():
    parser = argparse.ArgumentParser(
        description="Consolidate features and labels into paired HDF5 files."
    )
    parser.add_argument("-i", "--interaction_tsv", required=True)
    parser.add_argument("-d", "--db_path", required=True)
    parser.add_argument("-o", "--output_dir", required=True)
    parser.add_argument(
        "-f", "--filename", required=True, help="Base filename (e.g., 'train_data_v1')"
    )
    parser.add_argument("--phage_table", required=True)
    parser.add_argument("--host_table", required=True)
    parser.add_argument("--serotype_table", required=True)

    args = parser.parse_args()

    build_model_data(
        interaction_tsv=args.interaction_tsv,
        db_path=args.db_path,
        output_dir=args.output_dir,
        base_filename=args.filename,
        phage_table=args.phage_table,
        host_table=args.host_table,
        serotype_table=args.serotype_table,
    )


if __name__ == "__main__":
    main()
