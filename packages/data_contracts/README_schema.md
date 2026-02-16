# data_contracts

The `data_contracts` package defines the **schemas, primary keys, and partitioning rules** for datasets used in this pipeline.
These contracts serve as the canonical source of truth for downstream analysis, storage, and Parquet lake writes.

## Purpose

- Ensure **consistent schema enforcement** across all stages of the pipeline.
- Version and track **data structure changes** independently of pipeline parameters.
- Support safe **joins, queries, and analytics** across partitions and datasets.

## Structure

Each module corresponds to a conceptual dataset (lake) and typically contains:

- `*_SCHEMA`: dict mapping column names to expected data types.
- `*_PRIMARY_KEYS`: list of columns that uniquely identify rows.
- Optional partitioning or metadata constants.

Example:

```python
# data_contracts/protein_lake.py
PROTEIN_LAKE_SCHEMA = {
    "protein_name": "string",
    "organism_name": "string",
    "hmm_domain_name": "string",
    "annot": "string",
    "sequence": "string",
    "evalue": "float",
    "score": "float",
    "length": "int",
    # etc.
}

PROTEIN_LAKE_PRIMARY_KEYS = ["protein_name", "organism_name", "hmm_domain_name"]
