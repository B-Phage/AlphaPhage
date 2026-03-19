# data_contracts/variant_lake.py

VARIANT_LAKE_SCHEMA = {
    "organism_id": "string",
    "hmm_id": "string",
    "variant_id": "string",  # The tokenized cluster assignment (e.g., "102")
    "sequence_aligned": "string",  # The MSA sequence used as a reference gallery
    "radius": "float64",  # Max internal distance (or floor) for thresholding
    "clustering_version": "string",  # Metadata: e.g., 'red-none_clusters-16_mat-identity'
}

VARIANT_LAKE_PRIMARY_KEYS = ["organism_id", "hmm_id"]

# Note: While organism_id and hmm_id are the primary keys for identifying a
# specific hit, the "Inference Gallery" is typically constructed by grouping
# by hmm_id and clustering_version.
