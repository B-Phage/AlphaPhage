import os

import pandas as pd


def generate_phage_report(csv_path, output_html, top_host_num=3, threshold=0.5):
    # 1. Load the data
    df = pd.read_csv(csv_path)

    # 2. Parse the interaction_id: 'host__phage'
    # Splitting by the double underscore as requested
    df[["host", "phage"]] = df["interaction_id"].str.split("__", expand=True)

    # 3. Process the data
    # Sort by phage and then by probability_pos descending
    df_sorted = df.sort_values(by=["phage", "probability_pos"], ascending=[True, False])

    # Group by phage and take the top N
    report_data = []
    unique_phages = df_sorted["phage"].unique()

    for phage in unique_phages:
        phage_subset = df_sorted[df_sorted["phage"] == phage]

        # Filter by threshold and take top N
        top_matches = phage_subset[phage_subset["probability_pos"] >= threshold].head(top_host_num)

        # Extract host names
        hosts = top_matches["host"].tolist()

        # Pad with empty strings if fewer than top_host_num matches found
        while len(hosts) < top_host_num:
            hosts.append("")

        report_data.append([phage] + hosts)

    # 4. Create the final DataFrame for the report
    column_names = ["Phage"] + [f"Host {chr(97 + i)}" for i in range(top_host_num)]
    report_df = pd.DataFrame(report_data, columns=column_names)

    # 5. Generate HTML
    html_content = f"""
    <html>
    <head>
        <title>Phage-Host Interaction Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f4f7f6; }}
            table {{ border-collapse: collapse; width: 100%; background: white; }}
            th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
            th {{ background-color: #4CAF50; color: white; text-transform: uppercase; }}
            tr:hover {{ background-color: #f5f5f5; }}
            h2 {{ color: #333; }}
        </style>
    </head>
    <body>
        <h2>Phage Prediction Report</h2>
        <p>Threshold: {threshold} | Showing top {top_host_num} hosts</p>
        {report_df.to_html(index=False)}
    </body>
    </html>
    """

    with open(output_html, "w") as f:
        f.write(html_content)

    print(f"Report successfully generated at: {output_html}")


# --- Configuration ---
if __name__ == "__main__":
    # Path to your CSV file
    input_csv = "../databases/inference/output/final_predictions.csv"

    output_file = "../databases/best_host_per_phage_rep.html"

    generate_phage_report(
        csv_path=input_csv,
        output_html=output_file,
        top_host_num=3,  # Configurable number of host columns
        threshold=0.05,  # Probability threshold
    )
