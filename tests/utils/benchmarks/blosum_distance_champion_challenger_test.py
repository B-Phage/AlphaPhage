import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from variants.distance import (
    _distance_matrix_blosum62,
    _distance_matrix_blosum62_vectorized,
)

"""
Test the new vectorised implementation of the distance matrix against the old

Run at terminal with

export PYTHONPATH=$PYTHONPATH:$(pwd)/packages && python tests/utils/blosum_distance_champion_challenger_test.py
"""


def run_benchmark_and_plot(data_path: str, output_img: str = "benchmark_results_blosum62.png"):
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return

    # Load the challenging input
    alignment = np.load(data_path)
    n_total = alignment.shape[0]

    # Define scaling steps (subset sizes)
    # We use a logarithmic-ish scale to show the exponential growth of the loop version
    subsets = [10, 25, 50, 75, 100, 150, 200]
    subsets = [s for s in subsets if s <= n_total]

    results = []

    print(f"Starting Benchmark on {data_path}...")

    for n in subsets:
        subset = alignment[:n, :]

        # Benchmark Champion (Loops)
        t0 = time.perf_counter()
        _distance_matrix_blosum62(subset)
        t_loop = time.perf_counter() - t0

        # Benchmark Challenger (Vectorized)
        t1 = time.perf_counter()
        _distance_matrix_blosum62_vectorized(subset)
        t_vec = time.perf_counter() - t1

        results.append(
            {
                "Sequences": n,
                "Loop Version (s)": t_loop,
                "Vectorized Version (s)": t_vec,
                "Speedup": t_loop / t_vec if t_vec > 0 else 0,
            }
        )
        print(f"Done n={n}: Speedup {t_loop/t_vec:.2f}x")

    df = pd.DataFrame(results)

    # --- VISUALIZATION ---
    sns.set_theme(style="whitegrid")
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot Time on Primary Y-Axis
    sns.lineplot(
        data=df,
        x="Sequences",
        y="Loop Version (s)",
        label="Loop (Champion)",
        marker="o",
        ax=ax1,
        color="red",
    )
    sns.lineplot(
        data=df,
        x="Sequences",
        y="Vectorized Version (s)",
        label="Vectorized (Challenger)",
        marker="s",
        ax=ax1,
        color="blue",
    )
    ax1.set_ylabel("Execution Time (seconds)")
    ax1.set_title("MSA Distance Matrix: Loop vs Vectorized Performance")

    # Plot Speedup on Secondary Y-Axis
    ax2 = ax1.twinx()
    sns.lineplot(
        data=df,
        x="Sequences",
        y="Speedup",
        label="Speedup Factor",
        color="green",
        linestyle="--",
        ax=ax2,
    )
    ax2.set_ylabel("Speedup Factor (x-fold faster)")
    ax2.grid(False)

    # 1. Final Parity Check (DO THIS FIRST)
    # Use a fixed small sample (e.g., 50) so it's instant but reliable
    print("\nVerifying numerical parity on a subset...")
    check_size = min(len(alignment), 50)
    sample = alignment[:check_size]

    dist_old = _distance_matrix_blosum62(sample)
    dist_new = _distance_matrix_blosum62_vectorized(sample)

    parity = np.allclose(dist_old, dist_new, atol=1e-7)
    print(f"Numerical Parity Check (n={check_size}): {'PASSED' if parity else 'FAILED'}")

    # 2. Save and Show the Plot (DO THIS LAST)
    plt.tight_layout()
    plt.savefig(output_img)
    print(f"Report saved to: {output_img}")

    print("\nDisplaying plot... (Close the window to finish the script)")
    plt.show()


if __name__ == "__main__":
    run_benchmark_and_plot("/Users/tamsen/Code/AlphaPhage/tests/data/challenging_alignment.npy")

    print("\n--- BENCHMARK COMPLETE ---")
