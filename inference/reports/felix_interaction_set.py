# Inference validation report example for the Felixounavirus Clade

from external_db_provisioning.extract_interactions import extract_filtered_interactions
from inference_val_rep import generate_validation_report
from utils.config_loader import get_config


def felix_rep():

    felix_phages = [
        "LF110_P4",
        "NRG_11A2",
        "NRG_11B1",
        "MN342151",
        "ESh22",
        "LF110_P1",
        "AN17_P8",
        "SUSP1",
        "SUSP2",
        "LF110_P2",
        "LF82_P9",
        "LF82_P1",
        "LF82_P10",
        "LF82_P6",
        "LF82_P5",
        "LF82_P2",
        "LF110_P3",
        "LF82_P4",
        "LF82_P3",
    ]
    hosts = None  # Gets all hosts for these two phages
    felix_int = "../databases/felix_phage_set/interactions.tsv"

    config_path = "ml_assets/runtime_configs/interactions.yaml"

    cfg = get_config(override_path=config_path, params_set="data_intake")

    extract_filtered_interactions(
        source_db_path=cfg.source_db,
        output_tsv=felix_int,
        phage_list=felix_phages,
        host_list=hosts,
        table_name=getattr(cfg, "table_name", None),
    )

    predictions_path = "../databases/inference/output/final_predictions.csv"
    output_html = "../databases/felix_phage_set/val_rep.html"

    generate_validation_report(
        predictions_path=predictions_path, ground_truth_path=felix_int, output_html=output_html
    )


if __name__ == "__main__":
    felix_rep()
