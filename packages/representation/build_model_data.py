import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import utils.consol_log as log
from data_contracts.h5_schema import H5DataKeys
from storage_io.duckdb_client import DuckDBClient

logger = log.get_logger()


def build_model_data(
    interaction_tsv: str,
    db_path: str,
    output_dir: str,
    base_filename: str,
    phage_table: str,
    host_table: str,
    serotype_table: str,
):
    """
    Constructs the X (features) and CL (labels) HDF5 files.
    - x_{base_filename}.h5: Contains features and metadata info.
    - cl_{base_filename}.h5: Contains the interaction labels.
    """
    keys = H5DataKeys()

    # Construct Paths
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    x_file = out_path / f"x_{base_filename}.h5"
    cl_file = out_path / f"cl_{base_filename}.h5"

    # 1. Load interactions
    logger.info(f"📖 Loading interactions from {interaction_tsv}...")
    df_interact = pd.read_csv(interaction_tsv, sep="\t", index_col=0)
    pairs = df_interact.stack().reset_index()
    pairs.columns = ["host", "phage", "label"]
    pairs = pairs[pairs["label"] != -1].copy()

    # 2. Extract features and Metadata
    logger.info(f"🗄️ Querying features from {db_path}...")
    with DuckDBClient(db_path, read_only=True) as client:
        # Metadata for Versions
        phage_meta = client.execute(
            f"SELECT DISTINCT hmm_id, clustering_version FROM {phage_table}"
        ).df()
        host_meta = client.execute(
            f"SELECT DISTINCT hmm_id, clustering_version FROM {host_table}"
        ).df()

        # Pivot Phage
        phage_features = (
            client.execute(
                f"PIVOT {phage_table} ON hmm_id USING FIRST(token_id) GROUP BY organism_id"
            )
            .df()
            .fillna(0)
        )

        # Pivot Host + Join Serotypes
        host_query = f"""
            WITH wide_variants AS (PIVOT {host_table} ON hmm_id USING FIRST(token_id) GROUP BY organism_id)
            SELECT s.organism_id, s.k_type_dim1, s.k_type_dim2, v.* EXCLUDE (organism_id)
            FROM {serotype_table} s
            LEFT JOIN wide_variants v ON s.organism_id = v.organism_id
        """
        host_features = client.execute(host_query).df().fillna(0)

    # 3. Join Features to Pairs
    logger.info("🔗 Joining wide features with interaction labels...")
    data = pairs.merge(phage_features, left_on="phage", right_on="organism_id")
    data = data.merge(
        host_features, left_on="host", right_on="organism_id", suffixes=("_phage", "_host")
    )

    # 4. Prepare Metadata Info Arrays
    phage_feature_cols = [c for c in phage_features.columns if c != "organism_id"]
    p_ver_map = dict(zip(phage_meta["hmm_id"], phage_meta["clustering_version"]))
    phage_info = np.array(
        [phage_feature_cols, [str(p_ver_map.get(c, "unknown")) for c in phage_feature_cols]]
    ).astype("S100")

    hmm_cols = [
        c for c in host_features.columns if c not in ["organism_id", "k_type_dim1", "k_type_dim2"]
    ]
    h_ver_map = dict(zip(host_meta["hmm_id"], host_meta["clustering_version"]))
    host_info_row1 = ["k_type_dim1", "k_type_dim2"] + hmm_cols
    host_info_row2 = ["best_match", "best_match"] + [
        str(h_ver_map.get(c, "unknown")) for c in hmm_cols
    ]
    host_info = np.array([host_info_row1, host_info_row2]).astype("S100")

    # 5. Extract Numeric Arrays
    phage_vecs = data[phage_feature_cols].values.astype(np.int32)
    host_vecs = data[host_info_row1].values.astype(np.int32)
    labels = data["label"].values.astype(np.int32)
    pair_keys = data[["phage", "host"]].values.astype("S100")
    num_samples = len(data)
    indices = np.column_stack((np.arange(num_samples), np.arange(num_samples)))

    # 6. Writing Output Files

    # Save X (Features)
    logger.info(f"💾 Writing Feature file (X) to {x_file}...")
    with h5py.File(x_file, "w") as f:
        f.create_dataset(keys.PHAGE_VEC, data=phage_vecs)
        f.create_dataset(keys.HOST_VEC, data=host_vecs)
        f.create_dataset(keys.PAIR_KEYS, data=pair_keys)
        f.create_dataset(keys.PAIR_INDICES, data=indices)
        f.create_dataset(keys.PHAGE_VEC_INFO, data=phage_info)
        f.create_dataset(keys.HOST_VEC_INFO, data=host_info)

    # Save CL (Labels)
    logger.info(f"💾 Writing Label file (CL) to {cl_file}...")
    with h5py.File(cl_file, "w") as f:
        f.create_dataset(keys.PAIR_INT, data=labels)

    logger.info(f"✅ Training data split complete: {num_samples} samples.")


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
    parser.add_argument("--phage_table", default="phage_variants_v1")
    parser.add_argument("--host_table", default="bacteria_variants_v1")
    parser.add_argument("--serotype_table", default="bacteria_serotypes_v1")

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
