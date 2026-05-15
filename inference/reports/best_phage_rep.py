import pandas as pd


def generate_host_report(csv_path, output_html, top_phage_num=3, threshold=0.1):
    """
    Identifies and reports the most likely phages to infect a given list of hosts
    based on model prediction probabilities.

    This function acts as a filtering and visualization layer for model inference
    results. It allows the user to define the stringency of the report by setting
    an infectivity threshold and limits the output to a user-specified number of
    top phage candidates per host.

    Args:
        csv_path (str): Path to the CSV containing model predictions. Expected
            to have 'interaction_id' (host__phage) and 'probability_pos'.
        output_html (str): The file path for the generated HTML summary.
        top_phage_num (int, optional): The maximum number of phage candidates
            the user is willing to include in the report for each host.
            Defaults to 3.
        threshold (float, optional): The minimum infectivity probability
            score "good enough" to be considered a valid match.
            Defaults to 0.1.

    Returns:
        None: Saves a styled HTML table to the specified output path.
    """
    # 1. Load the data
    df = pd.read_csv(csv_path)

    # 2. Parse the interaction_id: 'host__phage'
    df[["host", "phage"]] = df["interaction_id"].str.split("__", expand=True)

    # 3. Process the data
    df_sorted = df.sort_values(by=["host", "probability_pos"], ascending=[True, False])

    report_data = []
    unique_hosts = df_sorted["host"].unique()

    for host in unique_hosts:
        host_subset = df_sorted[df_sorted["host"] == host]

        # Filter by threshold and take top N phages
        top_matches = host_subset[host_subset["probability_pos"] >= threshold].head(top_phage_num)

        phages = top_matches["phage"].tolist()

        # Pad with empty strings if fewer than top_phage_num matches found
        while len(phages) < top_phage_num:
            phages.append("")

        report_data.append([host] + phages)

    # 4. Create the final DataFrame
    column_names = ["Host"] + [f"Phage {chr(97 + i)}" for i in range(top_phage_num)]
    report_df = pd.DataFrame(report_data, columns=column_names)

    # 5. Generate HTML
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


if __name__ == "__main__":
    input_csv = "../databases/inference/output/final_predictions.csv"
    output_file = "../databases/host_top_phages_report.html"

    generate_host_report(
        csv_path=input_csv,
        output_html=output_file,
        top_phage_num=3,
        threshold=0.6,
    )
