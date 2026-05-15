from pathlib import Path

import utils.consol_log as log

# Import the helper functions from your existing logic file
# Replace 'your_logic_module' with the actual filename where these are defined
from hmm.inference_prep import (
    filter_dat_annotations,
    filter_phrog_annotations,
    prepare_inference_hmm_set,
)
from hmm.io import split_hmms
from utils.config_loader import get_config

logger = log.get_logger()


def snapshot_hmm_db(snapshot: str, new_hmm_db_name: str):
    """
    Creates a snapshot of the HMM databases and annotations for inference.

    Args:
        snapshot (str): The snapshot identifier/version string.
        new_hmm_db_name (str): The substitution string for path replacement (e.g., "/hmm_inf/").
    """
    logger.info(f"Starting HMM snapshot: {snapshot} with database tag: {new_hmm_db_name}")

    # Load configurations
    host_config_path = "ml_assets/runtime_configs/bac_preprocessing.yaml"
    phage_config_path = "ml_assets/runtime_configs/phage_preprocessing.yaml"

    host_hmm_cfg = get_config(override_path=host_config_path, params_set="hmm_config")
    host_msa_cfg = get_config(override_path=host_config_path, params_set="msa_config")
    phage_hmm_cfg = get_config(override_path=phage_config_path, params_set="hmm_config")
    phage_msa_cfg = get_config(override_path=phage_config_path, params_set="msa_config")

    # --- 1. PHROG (Phage) Processing ---
    logger.info("Processing PHROG database subset...")

    phage_unpacked_hmm_dir = phage_msa_cfg.split_hmm_dir
    phage_model_tsv = (
        f"../databases/inference/{snapshot}/reference/vector_info/phage_vec_1_info.tsv"
    )

    # Generate output path via replacement
    phage_output_path = phage_hmm_cfg.hmm_multifile.replace("/hmm/", new_hmm_db_name)

    # Build the pressed database
    prepare_inference_hmm_set(
        phage_unpacked_hmm_dir, phage_model_tsv, phage_output_path, as_pressed=True
    )

    # Split the new database into individual files for the inference environment
    phage_new_split = phage_unpacked_hmm_dir.replace("/hmm/", new_hmm_db_name)
    split_hmms(phage_output_path, phage_new_split)

    # Filter PHROG TSV annotations
    phage_input_tsv = phage_hmm_cfg.annotations
    phage_output_tsv = phage_input_tsv.replace("/hmm/", new_hmm_db_name)
    filter_phrog_annotations(phage_input_tsv, phage_output_tsv, phage_model_tsv)

    # --- 2. PFAM (Host) Processing ---
    logger.info("Processing PFAM database subset...")

    host_unpacked_hmm_dir = host_msa_cfg.split_hmm_dir
    host_model_tsv = f"../databases/inference/{snapshot}/reference/vector_info/host_vec_1_info.tsv"

    # Generate output path via replacement
    host_output_path = host_hmm_cfg.hmm_multifile.replace("/hmm/", new_hmm_db_name)

    # Build text-based database (as_pressed=False as per your original logic)
    prepare_inference_hmm_set(
        host_unpacked_hmm_dir, host_model_tsv, host_output_path, as_pressed=False
    )

    # Split the new database into individual files
    host_new_split = host_unpacked_hmm_dir.replace("/hmm/", new_hmm_db_name)
    split_hmms(host_output_path, host_new_split)

    # Filter PFAM .dat annotations
    host_input_dat = host_hmm_cfg.annotations
    host_output_dat = host_input_dat.replace("/hmm/", new_hmm_db_name)
    filter_dat_annotations(host_input_dat, host_output_dat, host_model_tsv)

    logger.info(f"Snapshot {snapshot} HMM databases successfully prepared.")
