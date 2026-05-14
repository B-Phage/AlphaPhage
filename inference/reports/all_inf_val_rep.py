# Inference validation report example for the Felixounavirus Clade

from external_db_provisioning.extract_interactions import extract_filtered_interactions
from inference_val_rep import generate_validation_report
from utils.config_loader import get_config


def all_val():

    phages = None  # Gets all phage
    hosts = None  # Gets all hosts for these two phages
    int_mat = "../databases/inference/output/all_interactions.tsv"

    config_path = "ml_assets/runtime_configs/interactions.yaml"

    cfg = get_config(override_path=config_path, params_set="data_intake")

    extract_filtered_interactions(
        source_db_path=cfg.source_db,
        output_tsv=int_mat,
        phage_list=phages,
        host_list=hosts,
        table_name=getattr(cfg, "table_name", None),
    )

    predictions_path = "../databases/inference/output/final_predictions.csv"
    output_html = "../databases/inference/output/val_rep.html"

    generate_validation_report(
        predictions_path=predictions_path, ground_truth_path=int_mat, output_html=output_html
    )


if __name__ == "__main__":
    all_val()
