import numpy as np
import pandas as pd
import plotly.graph_objects as go
import utils.consol_log as log
from data_contracts.interaction_db import InteractionTableContract
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

logger = log.get_logger()


def generate_validation_report(
    predictions_path: str, ground_truth_path: str, output_html: str, threshold: float = 0.5
):
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

    x_labels, y_labels = ["Predicted Neg", "Predicted Pos"], ["Actual Neg", "Actual Pos"]

    # --- 3. COMPONENT: Summary Table ---
    fig_table = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=["<b>Metric</b>", "<b>Value</b>"], fill_color="#f2f2f2", align="left"
                ),
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
                    ],
                    align="left",
                ),
            )
        ]
    )
    fig_table.update_layout(width=800, height=260, margin=dict(l=0, r=0, t=20, b=0))

    # --- 4. COMPONENT: Absolute Heatmap ---
    fig_abs = go.Figure(
        data=[
            go.Heatmap(
                z=cm_absolute,
                x=x_labels,
                y=y_labels,
                colorscale="Blues",
                text=[[str(v) for v in row] for row in cm_absolute],
                texttemplate="%{text}",
                showscale=False,
            )
        ]
    )
    fig_abs.update_layout(
        title="Confusion Matrix (Absolute)",
        xaxis_title="Prediction",
        yaxis_title="Truth",
        width=600,
        height=450,
        template="plotly_white",
        margin=dict(l=100, r=20, t=50, b=50),
    )

    # --- 5. COMPONENT: Proportional Heatmap ---
    rwg_scale = [[0, "rgb(215, 48, 39)"], [0.5, "rgb(255, 255, 255)"], [1, "rgb(26, 152, 80)"]]
    fig_prop = go.Figure(
        data=[
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
            )
        ]
    )
    fig_prop.update_layout(
        title="Confusion Matrix (Proportions)",
        xaxis_title="Prediction",
        yaxis_title="Truth",
        width=600,
        height=450,
        template="plotly_white",
        margin=dict(l=100, r=20, t=50, b=50),
    )

    # --- 6. GENERATE HTML SNIPPETS ---
    # include_plotlyjs='cdn' in the first call ensures the library loads before the plots draw
    table_html = fig_table.to_html(full_html=False, include_plotlyjs="cdn")
    abs_html = fig_abs.to_html(full_html=False, include_plotlyjs=False)
    prop_html = fig_prop.to_html(full_html=False, include_plotlyjs=False)

    # --- 7. ASSEMBLE FINAL FILE ---
    full_report = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8" />
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background-color: #f9f9f9; padding: 40px; }}
            .report-card {{ background: white; width: 850px; margin: auto; padding: 40px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
            .text-section {{ margin: 24px 0; line-height: 1.6; color: #34495e; }}
            .plot-container {{ margin: 20px 0; overflow: hidden; }}
            h1 {{ color: #2c3e50; margin-bottom: 30px; border-bottom: 2px solid #eee; padding-bottom: 10px; }}
            b {{ color: #2980b9; }}
        </style>
    </head>
    <body>
        <div class="report-card">
            <h1>AlphaPhage Validation Report</h1>

            <div class="text-section">
                <p>This report validates the predicted <b>AlphaPhage</b> phage-host interactions against known interactions in the database.</p>
                <p>The <b>threshold ({threshold})</b> is the value between 0 (not infect) and 1 (infect) where the prediction is
                classified or rounded as one or the other for evaluation purposes.</p>
                <p>Summary metrics provide a high-level evaluation of the validation. Because the dataset is usually quite skewed,
                one metric is usually not enough for proper validation.</p>
            </div>

            <div class="plot-container">{table_html}</div>

            <div class="text-section">
                <p>The <b>Absolute Matrix</b> shows the actual number of samples in each bucket and the skewness of the dataset is quite visible there.</p>
                <p>On the other hand, the <b>Proportional Matrix</b> shows the ratios of the predictions compared to the true number of infect / not infect and therefore some of the values overlap with the sensitivity and specificity percentages in the top summary table.</p>
            </div>

            <div class="plot-container">{abs_html}</div>
            <div class="plot-container">{prop_html}</div>
        </div>
    </body>
    </html>
    """

    with open(output_html, "w", encoding="utf-8") as f:
        f.write(full_report)

    logger.info(f"✅ Report successfully generated at {output_html}")
