# export KMP_DUPLICATE_LIB_OK=TRUE && conda run -n preprocessing python tests/utils/inspection/verify_modle_integ.py


import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix


def verify_model_integrity(predictions_csv_path, matrix_tsv_path, threshold=0.5):
    """
    Validates inference results against ground truth interaction matrix.

    Args:
        predictions_csv_path: Path to the 3-column CSV (interaction_id, prob_neg, prob_pos)
        matrix_tsv_path: Path to the wide ground-truth TSV (Host as index, Phages as columns)
        threshold: Classification threshold for probability_pos
    """
    # 1. Load the ground truth matrix and unpivot it
    # We assume the matrix has host as the first column/index
    gt_matrix = pd.read_csv(matrix_tsv_path, sep="\t", index_col=0)
    gt_long = gt_matrix.stack().reset_index()
    gt_long.columns = ["host", "phage", "label"]

    # Filter out unknown (-1) interactions
    gt_long = gt_long[gt_long["label"] != -1].copy()

    # Create a unique key that matches the prediction format: Host__Phage
    gt_long["interaction_id"] = gt_long["host"].astype(str) + "__" + gt_long["phage"].astype(str)

    # 2. Load the predictions
    preds_df = pd.read_csv(predictions_csv_path)

    # 3. Merge based on the interaction_id
    merged = pd.merge(preds_df, gt_long, on="interaction_id", how="inner")

    if merged.empty:
        print(
            "❌ CRITICAL ERROR: No matching interactions found between predictions and ground truth."
        )
        print("Check if the Host__Phage order is swapped or if naming conventions differ.")
        return

    # 4. Calculate Binary Predictions
    y_true = merged["label"].values
    y_pred = (merged["probability_pos"] >= threshold).astype(int)

    # 5. Calculate Metrics
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    print("-" * 30)
    print(f"Validation Results (N={len(merged)})")
    print("-" * 30)
    print(f"Confusion Matrix:")
    print(f"  TN: {tn} | FP: {fp}")
    print(f"  FN: {fn} | TP: {tp}")
    print("-" * 30)
    print(f"Accuracy:    {accuracy:.4f}")
    print(f"Sensitivity: {sensitivity:.4f} (True Positive Rate)")
    print(f"Specificity: {specificity:.4f} (True Negative Rate)")
    print("-" * 30)

    # Optional: Alert if performance is suspiciously bad
    if accuracy < 0.55:
        print("⚠️ WARNING: Accuracy is near random. Your Host/Phage features are likely swapped.")


if __name__ == "__main__":
    # Example usage:
    pred_csv = "/Users/tamsen/Code/databases/inference/output/final_predictions.csv"
    in_interact_tsv = (
        "/Users/tamsen/Code/AlphaPhage/ml_assets/data/processed/interactions/interaction_matrix.tsv"
    )
    verify_model_integrity(pred_csv, in_interact_tsv)
