# Hidden Markov Model (HMM) Package
This package handles the annotation of protein sets using HMM matching, providing the I/O layer between raw sequence data and downstream variant analysis.

## Pipeline Overview
The package manages the transition from raw protein sequences to filtered, high-confidence HMM annotations stored in a DuckDB-backed parquet lake.

## Key Components
- *Scanning:* Interfaces with pyhmmer to perform profile HMM searches.
- *Data Lake (`parquet_lake.py`):* Persists raw HMM hits partitioned by organism.
- *Summary Layer (`duckdb_io.py`):* Aggregates and filters results to identify the best representative protein for each HMM within an organism.

## The Filtering Logic (Summary Table)
The core of our annotation strategy is the "Summary Table." We don't just keep every HMM hit; we select the single best representative protein for each HMM within an organism.

We resolve two types of ambiguity:

1. HMM Competition: If one HMM hits multiple proteins, we select the top-scoring protein match.

2. Protein Overlap: If one protein hits multiple HMMs (overlapping domains), we prioritize the primary domain hit but allow secondary hits if they exceed a specific biological significance threshold (bit-score).

## Integration
The main entry point for the scanning pipeline is:
`workflows/cli/hmm_scan.py`

Note: This package requires the `preprocessing` conda environment.
