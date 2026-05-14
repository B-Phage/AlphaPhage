import pandas as pd


def generate_host_report(csv_path, output_html, top_phage_num=3, threshold=0.1):
    # 1. Load the data
    df = pd.read_csv(csv_path)

    # 2. Parse the interaction_id: 'host__phage'
    # Based on: Host name followed by "__" followed by the phage name
    df[["host", "phage"]] = df["interaction_id"].str.split("__", expand=True)

    # 3. Process the data
    # Sort by host alphabetically, then by probability_pos descending
    df_sorted = df.sort_values(by=["host", "probability_pos"], ascending=[True, False])

    report_data = []
    unique_hosts = df_sorted["host"].unique()

    for host in unique_hosts:
        host_subset = df_sorted[df_sorted["host"] == host]

        # Filter by threshold and take top N phages
        top_matches = host_subset[host_subset["probability_pos"] >= threshold].head(top_phage_num)

        # Extract phage names
        phages = top_matches["phage"].tolist()

        # Pad with empty strings if fewer than top_phage_num matches found
        while len(phages) < top_phage_num:
            phages.append("")

        report_data.append([host] + phages)

    # 4. Create the final DataFrame for the report
    # Columns are named "Phage a", "Phage b", etc.
    column_names = ["Host"] + [f"Phage {chr(97 + i)}" for i in range(top_phage_num)]
    report_df = pd.DataFrame(report_data, columns=column_names)

    # 5. Generate HTML with improved styling
    html_content = f"""
    <html>
    <head>
        <title>Host-Phage Interaction Report</title>
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 40px; background-color: #f8f9fa; }}
            table {{ border-collapse: collapse; width: 100%; background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
            th, td {{ padding: 15px; text-align: left; border-bottom: 1px solid #eee; }}
            th {{ background-color: #2c3e50; color: white; text-transform: uppercase; font-size: 0.9em; letter-spacing: 1px; }}
            tr:last-child td {{ border-bottom: none; }}
            tr:hover {{ background-color: #f1f4f6; }}
            h2 {{ color: #2c3e50; border-bottom: 2px solid #2c3e50; padding-bottom: 10px; }}
            .info {{ color: #666; margin-bottom: 20px; font-style: italic; }}
        </style>
    </head>
    <body>
        <h2>Best Phages per Host</h2>
        <p class="info">Threshold: Infect > {threshold} | Displaying top {top_phage_num} phages per host</p>
        {report_df.to_html(index=False)}
    </body>
    </html>
    """

    with open(output_html, "w") as f:
        f.write(html_content)

    print(f"Report successfully generated: {output_html}")


# --- Configuration ---
if __name__ == "__main__":
    # Path to your CSV file
    input_csv = "../databases/inference/output/final_predictions.csv"

    output_file = "../databases/host_top_phages_report.html"

    generate_host_report(
        csv_path=input_csv,
        output_html=output_file,
        top_phage_num=3,  # Set how many phage columns you want (a, b, c...)
        threshold=0.6,  # Minimum probability_pos to be included
    )
