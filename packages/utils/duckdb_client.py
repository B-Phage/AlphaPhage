from pathlib import Path
from typing import Any, Dict, Optional, Union

import duckdb
import pandas as pd


class DuckDBClient:
    def __init__(self, db_path: str = ":memory:", read_only: bool = False):
        """
        Initializes the connection.
        In production, use a persistent path like 'data/prod.duckdb'.
        """
        self.db_path = db_path
        # DuckDB connections are thread-safe and can be shared
        self.con = duckdb.connect(database=db_path, read_only=read_only)

    def run_query_file(
        self, file_path: Union[str, Path], replacements: Optional[Dict[str, str]] = None
    ) -> duckdb.DuckDBPyRelation:
        """
        Reads a .sql file and replaces placeholders like {table_name}.
        Uses DuckDB's .sql() which returns a lazy 'Relation' (no data moved yet).
        """
        sql = Path(file_path).read_text()

        if replacements:
            # We use standard Python string formatting for structural changes (tables/columns)
            # For data values, DuckDB's prepared statements are safer, but for
            # table names, formatting is necessary.
            sql = sql.format(**replacements)

        return self.con.sql(sql)

    def insert_dataframe(self, table_name: str, df: pd.DataFrame, primary_keys: list[str]):
        """
        Inserts a dataframe with 'Upsert' logic (Insert or Update).
        Avoids duplicates on critical keys using DuckDB's ON CONFLICT.
        """
        # Register the dataframe so DuckDB can see it as a table
        self.con.register("temp_df", df)

        # Determine the columns to update if a conflict occurs
        cols = [c for c in df.columns if c not in primary_keys]
        update_set = ", ".join([f"{c} = EXCLUDED.{c}" for c in cols])
        pk_str = ", ".join(primary_keys)

        # Production-grade Upsert logic
        sql = f"""
            INSERT INTO {table_name} SELECT * FROM temp_df
            ON CONFLICT ({pk_str})
            DO UPDATE SET {update_set}
        """
        self.con.execute(sql)
        self.con.unregister("temp_df")

    def join_tables(self, left_table: str, right_table: str, on_col: str, how: str = "inner"):
        """
        Helper for common joins. Returns a lazy relation.
        """
        return self.con.table(left_table).join(self.con.table(right_table), on_col, how=how)

    def close(self):
        self.con.close()
