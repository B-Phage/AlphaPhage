# data_contracts/variant_lake.py

VARIANT_LAKE_SCHEMA = {
    "organism_id": "string",
    "protein_id": "string",
    "variant_id": "int64",  # The cluster/token assignment
    "clustering_version": "string",  # Metadata to ensure version alignment
}

VARIANT_LAKE_PRIMARY_KEYS = ["organism_id", "protein_id"]
