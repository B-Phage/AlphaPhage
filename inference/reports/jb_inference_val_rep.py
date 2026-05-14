# Inference validation report example for the Felixounavirus Clade

from external_db_provisioning.extract_interactions import extract_filtered_interactions
from inference_val_rep import generate_validation_report
from utils.config_loader import get_config


def jb_rep():

    jb_phages = [
        "DIJ07_P2",
        "536_P9",
        "Mt1B1_P17",
        "DIJ07_P1",
        "536_P6",
        "536_P7",
        "536_P1",
    ]
    hosts = None  # Gets all hosts for these two phages
    felix_int = "../databases/jb_phage_set/interactions.tsv"

    config_path = "ml_assets/runtime_configs/interactions.yaml"

    cfg = get_config(override_path=config_path, params_set="data_intake")

    extract_filtered_interactions(
        source_db_path=cfg.source_db,
        output_tsv=felix_int,
        phage_list=jb_phages,
        host_list=hosts,
        table_name=getattr(cfg, "table_name", None),
    )

    predictions_path = "../databases/inference/output/final_predictions.csv"
    output_html = "../databases/felix_phage_set/jb_val_rep.html"

    generate_validation_report(
        predictions_path=predictions_path, ground_truth_path=felix_int, output_html=output_html
    )


if __name__ == "__main__":
    jb_rep()
