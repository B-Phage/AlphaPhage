# Inference validation report example for the Felixounavirus Clade

from external_db_provisioning.extract_interactions import extract_filtered_interactions
from inference_val_rep import generate_validation_report
from utils.config_loader import get_config


def vec_rep():

    vec_phages = [
        "DIJ06_P1",
        "Mt1B1_P10",
        "LI10_P6",
        "LI10_P5",
        "LI10_P4",
        "LI10_P3",
        "AL505_Sd2",
        "CrRp3",
        "NIC06_P3",
        "LF31_P3",
        "412_P2",
        "409_P6",
        "409_P5",
        "416_P4",
        "416_P5",
        "423_P1",
        "423_P10",
        "409_P3",
        "423_P7",
        "423_P9",
    ]
    hosts = None  # Gets all hosts for these two phages
    felix_int = "../databases/vec_phage_set/interactions.tsv"

    config_path = "ml_assets/runtime_configs/interactions.yaml"

    cfg = get_config(override_path=config_path, params_set="data_intake")

    extract_filtered_interactions(
        source_db_path=cfg.source_db,
        output_tsv=felix_int,
        phage_list=vec_phages,
        host_list=hosts,
        table_name=getattr(cfg, "table_name", None),
    )

    predictions_path = "../databases/inference/output/final_predictions.csv"
    output_html = "../databases/felix_phage_set/vec_val_rep.html"

    generate_validation_report(
        predictions_path=predictions_path, ground_truth_path=felix_int, output_html=output_html
    )


if __name__ == "__main__":
    vec_rep()
