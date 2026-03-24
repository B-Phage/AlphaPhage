import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import utils.consol_log as log
from data_contracts.h5_schema import TrainingManifest

# Import our new components
from representation.feature_extractor import FeatureExtractor
from representation.h5_writer import H5TrainingWriter
from representation.metadata_manager import MetadataManager
from storage_io.duckdb_client import DuckDBClient

logger = log.get_logger()


def load_interaction_pairs(path: str, include_unknown: bool = False) -> pd.DataFrame:
    """
    Unpivots the matrix.
    If include_unknown is False (Training), it drops -1.
    If include_unknown is True (Inference), it keeps everything.
    """
    df = pd.read_csv(path, sep="\t", index_col=0)
    pairs = df.stack().reset_index()
    pairs.columns = ["host", "phage", "label"]

    if not include_unknown:
        return pairs[pairs["label"] != -1].copy()
    return pairs


def align_features_from_list(
    df: pd.DataFrame, target_cols: List[str], missing_token: int
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Forces the DataFrame columns to match a specific list of strings.
    Pads missing HMMs with missing_token and drops HMMs not in target_cols.
    """
    # Reindex handles both ordering and padding (fillna)
    aligned_df = df.set_index("organism_id").reindex(columns=target_cols).fillna(missing_token)
    return aligned_df.reset_index(), target_cols


# --- Logic Subfunctions ---


def _extract_raw_features(
    db_path: str, phage_table: str, host_table: str, serotype_table: str, vocab_path: Optional[str]
):
    """Step 1: Get data out of DuckDB."""
    with DuckDBClient(db_path, read_only=True) as client:
        extractor = FeatureExtractor(client, vocab_path=vocab_path)
        return (
            extractor.get_wide_phage_features(phage_table),
            extractor.get_wide_host_features(host_table, serotype_table),
            extractor.get_metadata(phage_table),
            extractor.get_metadata(host_table),
            extractor.missing_token,
        )


def _align_all_features(
    phage_df: pd.DataFrame,
    host_df: pd.DataFrame,
    phage_ref_path: Optional[str],
    host_ref_path: Optional[str],
    m_token: int,
) -> Tuple[pd.DataFrame, List[str], pd.DataFrame, List[str]]:
    """Step 2 & 3: Align features to reference TSVs."""

    serotype_cols = ["k_type_dim1", "k_type_dim2"]

    # --- Phage Alignment ---
    if phage_ref_path and Path(phage_ref_path).exists():
        logger.info(f"🎯 Aligning phage features to reference: {phage_ref_path}")

        # Load reference and flatten to ensure we get all IDs
        ref_p = pd.read_csv(phage_ref_path, sep="\t", header=None, nrows=1)
        target_p_cols = ref_p.values.flatten().astype(str).tolist()

        # Force the dataframe to match these columns (padding with missing_token)
        phage_df, phage_cols = align_features_from_list(phage_df, target_p_cols, m_token)
    else:
        # Fallback: Just take whatever is in the DB
        phage_cols = [c for c in phage_df.columns if c != "organism_id"]

    # --- Host Alignment ---
    if host_ref_path and Path(host_ref_path).exists():
        logger.info(f"🎯 Aligning host features to reference: {host_ref_path}")

        # Load reference and flatten
        ref_h = pd.read_csv(host_ref_path, sep="\t", header=None, nrows=1)
        all_ref_names = ref_h.values.flatten().astype(str).tolist()

        # Separate the HMM IDs from the Serotype dimensions
        hmm_ref_cols = [c for c in all_ref_names if c not in serotype_cols]

        # Isolate HMMs, align them, then merge back with Serotypes
        hmm_only_df = host_df.drop(columns=[c for c in serotype_cols if c in host_df.columns])
        aligned_hmm_df, _ = align_features_from_list(hmm_only_df, hmm_ref_cols, m_token)

        # Re-attach Serotypes to the aligned HMM matrix
        host_df = host_df[["organism_id"] + serotype_cols].merge(aligned_hmm_df, on="organism_id")
        host_cols = serotype_cols + hmm_ref_cols
    else:
        # Fallback: Serotypes first, then everything else
        host_cols = serotype_cols + [
            c for c in host_df.columns if c not in ["organism_id"] + serotype_cols
        ]

    # Final Verification Log
    logger.info(
        f"📏 Final Feature Vector Dimensions -> Phage: {len(phage_cols)}, Host: {len(host_cols)}"
    )
    return phage_df, phage_cols, host_df, host_cols


def _validate_and_trim(
    interactions: pd.DataFrame, phage_df: pd.DataFrame, host_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Step 4: The intersection/filtering logic."""
    tsv_phages, tsv_hosts = set(interactions["phage"]), set(interactions["host"])
    feat_phages, feat_hosts = set(phage_df["organism_id"]), set(host_df["organism_id"])

    # Log missing
    m_p, m_h = tsv_phages - feat_phages, tsv_hosts - feat_hosts
    if m_p:
        logger.warning(f"⚠️ {len(m_p)} phages in interactions missing features: {list(m_p)[:5]}...")
    if m_h:
        logger.warning(f"⚠️ {len(m_h)} hosts in interactions missing features: {list(m_h)[:5]}...")

    # Filter interactions
    valid_data = interactions[
        interactions["phage"].isin(feat_phages) & interactions["host"].isin(feat_hosts)
    ].copy()
    if valid_data.empty:
        raise ValueError("❌ ZERO overlap found between interactions and features.")

    # Trim features to ONLY what is used in interactions
    final_p_ids, final_h_ids = set(valid_data["phage"]), set(valid_data["host"])
    logger.info(
        f"✂️ Trimmed unused: {len(feat_phages)-len(final_p_ids)} phages, {len(feat_hosts)-len(final_h_ids)} hosts."
    )

    trimmed_phage = phage_df[phage_df["organism_id"].isin(final_p_ids)].reset_index(drop=True)
    trimmed_host = host_df[host_df["organism_id"].isin(final_h_ids)].reset_index(drop=True)

    return valid_data, trimmed_phage, trimmed_host


# --- Main Orchestrator ---


def build_model_data(
    interaction_tsv: str,
    db_path: str,
    output_dir: str,
    base_filename: str,
    phage_table: str,
    host_table: str,
    serotype_table: str,
    phage_ref_path: Optional[str] = None,
    host_ref_path: Optional[str] = None,
    vocab_path: Optional[str] = None,
    is_inference: bool = False,
) -> None:
    """Orchestrates the construction of paired HDF5 datasets for model training or inference.

    This function performs a full ETL pipeline: extracting raw features from DuckDB,
    aligning them against reference schemas, filtering interaction pairs based on
    feature availability, and trimming unused features to minimize the output file size.

    The final HDF5 file contains compressed integer-encoded feature vectors for phages
    and hosts, interaction labels, and an index mapping for reconstruction.

    Args:
        interaction_tsv: Path to the TSV file containing 'phage', 'host', and 'label' columns.
        db_path: Path to the DuckDB database file containing feature tables.
        output_dir: Directory where the resulting HDF5 and metadata files will be saved.
        base_filename: Prefix used for the generated output files.
        phage_table: Name of the table in DuckDB containing phage HMM features.
        host_table: Name of the table in DuckDB containing host HMM features.
        serotype_table: Name of the table in DuckDB containing host serotype/K-type data.
        phage_ref_path: Optional; path to a reference TSV to enforce a specific phage
            feature schema/ordering.
        host_ref_path: Optional; path to a reference TSV to enforce a specific host
            feature schema/ordering.
        vocab_path: Optional; path to a YAML file for vocabulary mapping in the extractor.
        is_inference: If True, retains rows with unknown labels (-1). If False,
            filters for known interactions only.

    Raises:
        ValueError: If no overlap exists between the organisms in the interaction TSV
            and the organisms available in the feature database.
        FileNotFoundError: If the provided database path or interaction TSV does not exist.

    Note:
        The function logs warnings for any organisms found in the interaction TSV that
        are missing from the feature database. These organisms are excluded from the
        final HDF5 output.
    """

    # 1. Extraction
    phage_df, host_df, p_meta, h_meta, m_token = _extract_raw_features(
        db_path, phage_table, host_table, serotype_table, vocab_path
    )

    # 2 & 3. Alignment
    phage_df, p_cols, host_df, h_cols = _align_all_features(
        phage_df, host_df, phage_ref_path, host_ref_path, m_token
    )

    # 4. Interaction Mapping & Trimming
    interactions = load_interaction_pairs(interaction_tsv, include_unknown=is_inference)
    data, phage_df, host_df = _validate_and_trim(interactions, phage_df, host_df)

    # Create Integer Mapping
    p_map = {id_: i for i, id_ in enumerate(phage_df["organism_id"])}
    h_map = {id_: i for i, id_ in enumerate(host_df["organism_id"])}

    # Use np.asarray() to explicitly convert Pandas/Extension types to pure NumPy
    h_idx = np.asarray(data["host"].map(h_map), dtype="int32")
    p_idx = np.asarray(data["phage"].map(p_map), dtype="int32")

    # Use a list of explicit ndarrays
    pair_indices = np.stack([h_idx, p_idx], axis=1)

    # 5. Write HDF5
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    writer = H5TrainingWriter(out_path, base_filename)

    # Pass variables directly. No dict keys to misspell!
    x_file, cl_file = writer.write(
        phage_vecs=np.asarray(phage_df[p_cols], dtype="int32"),
        host_vecs=np.asarray(host_df[h_cols], dtype="int32"),
        pair_indices=pair_indices,
        pair_keys=np.asarray(data[["phage", "host"]], dtype="S100"),
        labels=np.asarray(data["label"], dtype="int32"),
        phage_info=MetadataManager.build_info_array(p_cols, p_meta),
        host_info=MetadataManager.build_info_array(h_cols, h_meta),
    )

    # 6. Optional: Log the Manifest stats (matches TrainingManifest contract)
    manifest = TrainingManifest(
        phage_dim=len(p_cols),
        host_dim=len(h_cols),
        num_samples=len(data),
        phage_table=phage_table,
        host_table=host_table,
        serotype_table=serotype_table,
        vocab_hash="hash_placeholder",  # You can implement a real hash here if needed
    )

    logger.info(f"✅ Created HDF5: {x_file.name}")
    logger.info(
        f"📊 Manifest: {manifest.num_samples} samples, {manifest.phage_dim}p x {manifest.host_dim}h features."
    )


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
    parser.add_argument("--phage_ref", help="Path to phage_info.tsv reference")
    parser.add_argument("--host_ref", help="Path to host_info.tsv reference")
    parser.add_argument("-v", "--vocab", help="Path to vocab.yaml")

    # Inference Mode Flag
    parser.add_argument(
        "--inference", action="store_true", help="If set, keeps unknown labels (-1) for prediction"
    )

    args = parser.parse_args()

    build_model_data(
        interaction_tsv=args.interaction_tsv,
        db_path=args.db_path,
        output_dir=args.output_dir,
        base_filename=args.filename,
        phage_table=args.phage_table,
        host_table=args.host_table,
        serotype_table=args.serotype_table,
        phage_ref_path=args.phage_ref,  # Will be None if flag is missing
        host_ref_path=args.host_ref,  # Will be None if flag is missing
        vocab_path=args.vocab,
        is_inference=args.inference,
    )


if __name__ == "__main__":
    main()
