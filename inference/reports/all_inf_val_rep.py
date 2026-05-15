"""
Inference Validation Module (Full Database)

This script performs a comprehensive validation of model predictions against the
entire ground-truth inference set. Unlike subset-specific reports, this
checks every interaction available in the database.

Requirements:
    - A local DVC checkout of the database is required to pull ground-truth data.
    - Not suitable for lightweight Docker deployments (requires full DB access).
    - Environment: 'preprocessing' conda environment.
"""

from external_db_provisioning.extract_interactions import extract_filtered_interactions
from inference_val_rep import generate_validation_report
from utils.config_loader import get_config


def validate_all():
    """
    Executes the full validation pipeline:
    1. Extracts ground-truth interaction matrices for all hosts and phages.
    2. Compares model predictions against the ground-truth.
    3. Generates a comprehensive HTML validation report.
    """

    phages = None  # None selects all available phages
    hosts = None  # None selects all available hosts
    int_mat = "../databases/inference/output/all_interactions.tsv"

    config_path = "ml_assets/runtime_configs/interactions.yaml"
    cfg = get_config(override_path=config_path, params_set="data_intake")

    # Extract ground truth from the source database defined in config
    extract_filtered_interactions(
        source_db_path=cfg.source_db,
        output_tsv=int_mat,
        phage_list=phages,
        host_list=hosts,
        table_name=getattr(cfg, "table_name", None),
    )

    predictions_path = "../databases/inference/output/final_predictions.csv"
    output_html = "../databases/inference/output/val_rep.html"

    # Generate the final visual comparison report
    generate_validation_report(
        predictions_path=predictions_path, ground_truth_path=int_mat, output_html=output_html
    )


if __name__ == "__main__":
    validate_all()
