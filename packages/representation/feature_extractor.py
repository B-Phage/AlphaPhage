import pandas as pd
from storage_io.duckdb_client import DuckDBClient
from utils.config_loader import get_vocab


class FeatureExtractor:
    def __init__(self, client: DuckDBClient):
        self.client = client
        vocab = get_vocab()
        self.missing_token = vocab.vocab_clusters_single.missing

    def get_wide_phage_features(self, table_name: str) -> pd.DataFrame:
        query = f"PIVOT {table_name} ON hmm_id USING FIRST(token_id) GROUP BY organism_id"
        return self.client.execute(query).df().fillna(self.missing_token)

    def get_wide_host_features(self, host_table: str, serotype_table: str) -> pd.DataFrame:
        query = f"""
            WITH wide_variants AS (PIVOT {host_table} ON hmm_id USING FIRST(token_id) GROUP BY organism_id)
            SELECT s.organism_id, s.k_type_dim1, s.k_type_dim2, v.* EXCLUDE (organism_id)
            FROM {serotype_table} s
            LEFT JOIN wide_variants v ON s.organism_id = v.organism_id
        """
        return self.client.execute(query).df().fillna(self.missing_token)

    def get_metadata(self, table_name: str) -> pd.DataFrame:
        return self.client.execute(
            f"SELECT DISTINCT hmm_id, clustering_version FROM {table_name}"
        ).df()
