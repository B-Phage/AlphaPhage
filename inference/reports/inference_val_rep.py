import numpy as np
import pandas as pd
import plotly.graph_objects as go
import utils.consol_log as log
from data_contracts.interaction_db import InteractionTableContract
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

logger = log.get_logger()


def generate_validation_report(
    predictions_path: str, ground_truth_path: str, output_html: str, threshold: float = 0.5
):
    """
    Compares model predictions against DB ground truth and generates an HTML report
    with a fixed aspect ratio and localized color bars.
    """
    contract = InteractionTableContract()

    # 1. Load and Merge Data
    preds_df = pd.read_csv(predictions_path)
    truth_df = pd.read_csv(ground_truth_path, sep="\t")
    preds_df[[contract.HOST, contract.PHAGE]] = preds_df["interaction_id"].str.split(
        "__", expand=True
    )

    merged = pd.merge(preds_df, truth_df, on=[contract.HOST, contract.PHAGE], how="inner").dropna(
        subset=[contract.INTERACTION]
    )

    if merged.empty:
        logger.error("❌ No overlapping interactions found.")
        return

    # 2. Metrics Calculation
    y_true = merged[contract.INTERACTION].astype(int)
    y_pred = (merged["probability_pos"] >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    cm_absolute = np.array([[tn, fp], [fn, tp]])
    cm_proportions = cm_absolute.astype("float") / cm_absolute.sum(axis=1)[:, np.newaxis]
    cm_proportions = np.nan_to_num(cm_proportions)

    # 3. Create Subplots
    fig = make_subplots(
        rows=3,
        cols=1,
        row_heights=[0.2, 0.4, 0.4],
        vertical_spacing=0.1,
        subplot_titles=(
            "Summary Metrics",
            "Confusion Matrix (Absolute)",
            "Confusion Matrix (Proportions)",
        ),
        specs=[[{"type": "table"}], [{"type": "xy"}], [{"type": "xy"}]],
    )

    # 4. Table
    fig.add_trace(
        go.Table(
            header=dict(values=["<b>Metric</b>", "<b>Value</b>"], fill_color="lightgrey"),
            cells=dict(
                values=[
                    ["Recall", "Specificity", "Precision", "F1", "Samples"],
                    [
                        f"{recall_score(y_true, y_pred):.4f}",
                        f"{tn/(tn+fp):.4f}",
                        f"{precision_score(y_true, y_pred):.4f}",
                        f"{f1_score(y_true, y_pred):.4f}",
                        len(merged),
                    ],
                ]
            ),
        ),
        row=1,
        col=1,
    )

    # 5. Absolute Heatmap
    x_labels, y_labels = ["Predicted Neg", "Predicted Pos"], ["Actual Neg", "Actual Pos"]

    fig.add_trace(
        go.Heatmap(
            z=cm_absolute,
            x=x_labels,
            y=y_labels,
            colorscale="Blues",
            text=[[str(v) for v in row] for row in cm_absolute],
            texttemplate="%{text}",
            showscale=False,
        ),
        row=2,
        col=1,
    )

    # 6. Proportional Heatmap (Fixed Colorbar)
    rwg_scale = [[0, "rgb(215, 48, 39)"], [0.5, "rgb(255, 255, 255)"], [1, "rgb(26, 152, 80)"]]

    fig.add_trace(
        go.Heatmap(
            z=cm_proportions,
            x=x_labels,
            y=y_labels,
            colorscale=rwg_scale,
            zmin=0,
            zmax=1,
            text=[[f"{v:.2%}" for v in row] for row in cm_proportions],
            texttemplate="%{text}",
            showscale=True,
            # Position colorbar specifically next to row 3
            colorbar=dict(
                title="Proportion",
                thickness=15,
                len=0.35,  # Length relative to whole plot (matches row_height 0.4)
                y=0.2,  # Centers it vertically on the 3rd row
                yanchor="middle",
            ),
        ),
        row=3,
        col=1,
    )

    # 7. Layout and Aspect Ratio
    fig.update_layout(
        title_text=f"<b>Validation Report (Threshold: {threshold})</b>",
        width=750,
        height=1000,
        autosize=False,
        template="plotly_white",
        margin=dict(l=60, r=60, t=100, b=60),
    )

    fig.update_xaxes(title_text="Prediction", row=2, col=1)
    fig.update_yaxes(title_text="Truth", row=2, col=1)
    fig.update_xaxes(title_text="Prediction", row=3, col=1)
    fig.update_yaxes(title_text="Truth", row=3, col=1)

    fig.write_html(output_html)
    logger.info(f"✅ Report saved to {output_html}")
