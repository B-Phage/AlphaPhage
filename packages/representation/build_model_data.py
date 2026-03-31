import argparse
from pathlib import Path

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
    interaction_tsv, db_path, output_dir, base_filename, phage_table, host_table, serotype_table
):
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
        Orchestrates the construction of paired HDF5 training datasets.

        This function integrates host and phage genomic features from a DuckDB
        repository with phenotypic interaction labels from a TSV matrix. It
        pivots long-format variant data into wide-format feature vectors and
        aligns them into two distinct HDF5 files (X for features, CL for labels)
        compatible with the AlphaPhage Transformer dataloader.

        Args:
            interaction_tsv: Path to the tab-separated matrix containing
                ground-truth host-phage interactions.
            db_path: Path to the persistent DuckDB file containing feature tables.
            output_dir: Directory where the resulting HDF5 files will be saved.
            base_filename: The core name used for output files; will be prefixed
                with 'x_' for features and 'cl_' for labels.
            phage_table: Name of the DuckDB table containing phage variants.
            host_table: Name of the DuckDB table containing host variants.
            serotype_table: Name of the DuckDB table containing host serotypes.

        Returns:
            None

        Raises:
            FileNotFoundError: If the interaction_tsv or db_path does not exist.
            KeyError: If the required tables or columns are missing from the
                DuckDB database.
            ValueError: If the join between interactions and features results
                in an empty dataset.

        Example:
            >>> build_model_data(
            ...     interaction_tsv="data/matrix.tsv",
            ...     db_path="data/features.db",
            ...     output_dir="data/processed",
            ...     base_filename="v1_run",
            ...     phage_table="phage_vars",
            ...     host_table="host_vars",
            ...     serotype_table="host_serotypes"
            ... )
            # Generates: x_v1_run.h5 and cl_v1_run.h5
        """

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Extraction (DuckDB)
    with DuckDBClient(db_path, read_only=True) as client:
        extractor = FeatureExtractor(client)
        phage_df = extractor.get_wide_phage_features(phage_table)
        host_df = extractor.get_wide_host_features(host_table, serotype_table)
        p_meta = extractor.get_metadata(phage_table)
        h_meta = extractor.get_metadata(host_table)

    # 2. Alignment (Pandas)
    logger.info("🔗 Aligning features with labels...")
    interactions = load_interaction_pairs(interaction_tsv)
    data = interactions.merge(phage_df, left_on="phage", right_on="organism_id")
    data = data.merge(host_df, left_on="host", right_on="organism_id", suffixes=("_p", "_h"))

    # 3. Metadata Preparation
    phage_cols = [c for c in phage_df.columns if c != "organism_id"]
    hmm_cols = [
        c for c in host_df.columns if c not in ["organism_id", "k_type_dim1", "k_type_dim2"]
    ]

    phage_info = MetadataManager.build_info_array(phage_cols, p_meta)
    host_info = MetadataManager.build_info_array(
        cols=["k_type_dim1", "k_type_dim2"] + hmm_cols,
        meta_df=h_meta,
        source_map={"k_type_dim1": "best_match", "k_type_dim2": "best_match"},
    )

    # 4. Final Output (HDF5)
    writer = H5TrainingWriter(out_path, base_filename)
    payload = {
        "phage_vecs": data[phage_cols].values.astype("int32"),
        "host_vecs": data[["k_type_dim1", "k_type_dim2"] + hmm_cols].values.astype("int32"),
        "labels": data["label"].values.astype("int32"),
        "pair_keys": data[["phage", "host"]].values.astype("S100"),
    }

    x_file, cl_file = writer.write(payload, phage_info, host_info)
    logger.info(f"✅ Created training set: {x_file.name} and {cl_file.name}")


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
