import os
from pathlib import Path

import pandas as pd
import pyhmmer.hmmer
import pyhmmer.plan7
import utils.consol_log as log
from hmm.io import split_hmms
from utils.config_loader import get_config

logger = log.get_logger()


def prepare_inference_hmm_set(
    unpacked_hmm_dir: str | Path,
    model_tsv: str | Path,
    output_path: str | Path,
    as_pressed: bool = True,
):
    """
    Consolidates a subset of HMMs from an unpacked directory into a single
    database file (.h3m or .hmm) based on a TSV selection.
    """
    # 1. Get the set of names to keep
    try:
        selected_set = get_selected_hmm_set(model_tsv)
    except Exception as e:
        logger.error(f"Failed to extract HMM set from TSV: {e}")
        raise

    logger.info(f"Preparing subset: {len(selected_set)} unique HMMs requested.")

    # 2. Collect HMM objects from the unpacked folder
    hmms_to_package = []
    missing_count = 0
    source_dir = Path(unpacked_hmm_dir)

    for name in selected_set:
        hmm_file_path = source_dir / f"{name}.hmm"

        if hmm_file_path.exists():
            with pyhmmer.plan7.HMMFile(hmm_file_path) as hmm_file:
                # read() extracts the HMM object from the individual file
                hmms_to_package.append(hmm_file.read())
        else:
            missing_count += 1
            logger.warning(f"Requested HMM '{name}' not found in {unpacked_hmm_dir}")

    if not hmms_to_package:
        logger.error("No HMMs were successfully loaded from the directory. Aborting.")
        return

    # 3. Handle directory creation and path normalization
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # 4. Output to the requested format
    if as_pressed:
        # Create the .h3m, .h3p, .h3f, and .h3i files
        output_prefix = str(out.with_suffix(""))
        pyhmmer.hmmer.hmmpress(hmms_to_package, output_prefix)
        logger.info(f"Pressed database created: {output_prefix}.h3m")
    else:
        # Save as a single standard monolithic HMM text file
        output_file = out.with_suffix(".hmm")
        with open(output_file, "wb") as dst:
            for hmm in hmms_to_package:
                hmm.write(dst)
        logger.info(f"Text database created: {output_file}")

    if missing_count > 0:
        logger.warning(
            f"Process complete. {len(hmms_to_package)} packaged, {missing_count} missing."
        )
    else:
        logger.info(f"Process complete. All {len(hmms_to_package)} HMMs packaged successfully.")


def filter_dat_annotations(
    input_dat: str | Path, output_dat: str | Path, model_tsv: str | Path
) -> int:
    """
    Filters a .dat annotation file based on HMM IDs defined in a TSV header.

    Args:
        input_dat: Path to the large source .dat file.
        output_dat: Path to save the filtered .dat file.
        model_tsv: Path to the TSV file containing the selected HMM names in the header.

    Returns:
        int: Total number of records written to the output file.
    """
    # 1. Get the set of names to keep using the helper function
    try:
        keep_names = get_selected_hmm_set(model_tsv)
    except Exception as e:
        logger.error(f"Could not retrieve HMM set for annotation filtering: {e}")
        raise

    # 2. Ensure output directory exists
    out_path = Path(output_dat)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    records_written = 0
    current_block = []
    current_id = None

    # 3. Stream through the file to keep memory usage low
    with (
        open(input_dat, "r", encoding="utf-8") as f_in,
        open(out_path, "w", encoding="utf-8") as f_out,
    ):

        for line in f_in:
            current_block.append(line)

            # Identify the record ID (e.g., #=GF ID   1-cysPrx_C)
            if line.startswith("#=GF ID"):
                parts = line.strip().split()
                # Index 2 contains the ID string
                if len(parts) >= 3:
                    current_id = parts[2]

            # If we hit the end-of-record marker, decide whether to write it
            if line.strip() == "//":
                if current_id in keep_names:
                    f_out.writelines(current_block)
                    records_written += 1

                # Reset buffers for the next block
                current_block = []
                current_id = None

    logger.info(f"Filtered .dat file complete: {records_written} records saved to {output_dat}.")
    return records_written


def get_selected_hmm_set(model_tsv: str | Path) -> set[str]:
    """
    Extracts HMM names from the first row of a TSV file.

    Args:
        model_tsv: Path to the TSV file.

    Returns:
        set: A set of HMM names for O(1) lookup.
    """
    path = Path(model_tsv)
    if not path.exists():
        raise FileNotFoundError(f"Model TSV not found at: {path}")

    with open(path, "r", encoding="utf-8") as f:
        first_line = f.readline().strip()

    if not first_line:
        return set()

    # Split by tab and convert to set
    selected_set = {name for name in first_line.split("\t") if name}

    return selected_set


def filter_phrog_annotations(
    input_tsv: str | Path, output_tsv: str | Path, model_tsv: str | Path
) -> int:
    """
    Filters PHROG annotations based on composite keys in model_tsv.

    Converts model_tsv keys (e.g., 'phrog_1') back to raw values (e.g., 1)
    to match the 'phrog' column in the annotation file.
    """
    # 1. Get the set of composite names (e.g., {'phrog_1', 'phrog_2'})
    try:
        selected_composite_set = get_selected_hmm_set(model_tsv)
    except Exception as e:
        logger.error(f"Could not retrieve HMM set: {e}")
        raise

    # 2. Extract the raw IDs from the composite strings
    # 'phrog_1' -> 1 (as integer or string depending on TSV types)
    # We strip 'phrog_' from the start and keep the rest.
    target_ids = []
    for item in selected_composite_set:
        if "_" in item:
            # Splits 'phrog_1' into ['phrog', '1'] and takes index 1
            target_ids.append(item.split("_", 1)[1])
        else:
            logger.warning(f"Unexpected HMM name format: {item}. Skipping.")

    # 3. Load the annotation TSV
    # We load IDs as strings to ensure matching works regardless of leading zeros
    df = pd.read_csv(input_tsv, sep="\t", dtype={"phrog": str})

    # 4. Filter the dataframe
    # Only keep rows where the 'phrog' column value is in our target_ids list
    filtered_df = df[df["phrog"].isin(target_ids)]

    # 5. Save the output
    out_path = Path(output_tsv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    filtered_df.to_csv(out_path, sep="\t", index=False)

    logger.info(f"Filtered TSV complete: {len(filtered_df)} PHROG records saved to {output_tsv}.")
    return len(filtered_df)


if __name__ == "__main__":

    snapshot = "TransformerX_v1_ep27_20260401"
    new_hmm_db_name = "/hmm_inf/"

    host_config_path = "ml_assets/runtime_configs/bac_preprocessing.yaml"
    phage_config_path = "ml_assets/runtime_configs/phage_preprocessing.yaml"
    host_hmm_cfg = get_config(override_path=host_config_path, params_set="hmm_config")
    host_msa_cfg = get_config(override_path=host_config_path, params_set="msa_config")
    phage_hmm_cfg = get_config(override_path=phage_config_path, params_set="hmm_config")
    phage_msa_cfg = get_config(override_path=phage_config_path, params_set="msa_config")

    # PHROG
    unpacked_hmm_dir = phage_msa_cfg.split_hmm_dir
    model_tsv = f"../databases/inference/{snapshot}/reference/vector_info/phage_vec_1_info.tsv"
    output_path = phage_hmm_cfg.hmm_multifile
    output_path = output_path.replace("/hmm/", new_hmm_db_name)

    output_path = "../databases/hmm_inf/pharokka_v1.4.0_databases/all_phrogs.h3m"
    as_pressed = True
    prepare_inference_hmm_set(unpacked_hmm_dir, model_tsv, output_path, as_pressed)
    new_split = unpacked_hmm_dir.replace("/hmm/", new_hmm_db_name)
    split_hmms(output_path, new_split)

    input_tsv = phage_hmm_cfg.annotations
    output_tsv = input_tsv.replace("/hmm/", new_hmm_db_name)
    filter_phrog_annotations(input_tsv, output_tsv, model_tsv)

    # PFAM
    unpacked_hmm_dir = host_msa_cfg.split_hmm_dir
    host_model_tsv = f"../databases/inference/{snapshot}/reference/vector_info/host_vec_1_info.tsv"
    output_path = host_hmm_cfg.hmm_multifile
    output_path = output_path.replace("/hmm/", new_hmm_db_name)
    as_pressed = False
    prepare_inference_hmm_set(unpacked_hmm_dir, host_model_tsv, output_path, as_pressed)
    new_split = unpacked_hmm_dir.replace("/hmm/", new_hmm_db_name)
    split_hmms(output_path, new_split)

    input_dat = host_hmm_cfg.annotations
    output_dat = input_dat.replace("/hmm/", new_hmm_db_name)
    filter_dat_annotations(input_dat, output_dat, host_model_tsv)
