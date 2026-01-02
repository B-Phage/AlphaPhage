import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Union

import duckdb
import pandas as pd


class DuckDBClient:
    """
    Thin, SQL-first DuckDB client with:
    - lazy relations
    - safe upserts
    - transaction support
    - light metadata helpers
    """

    def __init__(
        self,
        db_path: str = ":memory:",
        read_only: bool = False,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.db_path = db_path
        self.con = duckdb.connect(
            database=db_path,
            read_only=read_only,
            config=config or {},
        )

    # ------------------------------------------------------------------
    # Context management
    # ------------------------------------------------------------------

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def close(self):
        self.con.close()

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _quote_ident(name: str) -> str:
        """Safely quote an SQL identifier for DuckDB.

        Wraps the given table or column name in double quotes to prevent SQL syntax
        errors or accidental SQL injection. Raises a ValueError if the identifier
        contains a double quote.

        Args:
            name: The table or column name to quote.

        Returns:
            The safely quoted identifier string, e.g., '"my_table"'.

        Raises:
            ValueError: If the identifier contains a double quote character (`"`).
        """
        if '"' in name:
            raise ValueError(f"Invalid identifier: {name}")
        return f'"{name}"'

    @contextmanager
    def transaction(self):
        try:
            self.con.execute("BEGIN")
            yield
            self.con.execute("COMMIT")
        except Exception:
            self.con.execute("ROLLBACK")
            raise

    # ------------------------------------------------------------------
    # Core query helpers
    # ------------------------------------------------------------------

    def run_query_file(
        self,
        file_path: Union[str, Path],
        replacements: Optional[Dict[str, str]] = None,
    ) -> duckdb.DuckDBPyRelation:
        """
        Execute a SQL file and return a lazy DuckDB relation.

        Note:
        - Structural changes (tables / columns) must be done via `replacements`
        - Parameter binding is intentionally not supported here
        """
        sql = Path(file_path).read_text()

        if replacements:
            sql = sql.format(**replacements)

        return self.con.sql(sql)

    def execute(self, sql: str, params: Optional[tuple] = None) -> duckdb.DuckDBPyConnection:
        """
        Executes raw SQL and returns the connection object (materialized).
        """
        return self.con.execute(sql, params)

    def table(self, table_name: str) -> duckdb.DuckDBPyRelation:
        """
        Returns a lazy DuckDB relation for an existing table.

        This method provides a safe, SQL-first way to reference a table as a
        `DuckDBPyRelation`, enabling lazy query construction and composition
        without executing the query immediately.

        The table name is automatically quoted to prevent SQL injection and to
        support reserved keywords or mixed-case identifiers.

        Args:
            table_name: Name of the table to reference.

        Returns:
            A lazy `DuckDBPyRelation` representing the table.

        Raises:
            ValueError: If the table name contains invalid characters.
        """
        return self.con.table(self._quote_ident(table_name))

    # ------------------------------------------------------------------
    # DataFrame ingestion
    # ------------------------------------------------------------------

    def insert_dataframe(
        self,
        table_name: str,
        df: pd.DataFrame,
        primary_keys: Optional[Iterable[str]] = None,
    ) -> None:
        """
        Inserts or upserts a pandas DataFrame into an existing DuckDB table.

        If the table has primary keys (either specified via `primary_keys` or
        inferred from metadata), performs an upsert using `INSERT ... ON CONFLICT`.
        Non-primary-key columns are updated if a conflict occurs. If the table has
        no primary keys, performs a plain insert.

        All DataFrame columns must already exist in the target table.

        The operation is a no-op if the DataFrame is empty.

        Args:
            table_name: Name of the target table.
            df: Pandas DataFrame containing rows to insert.
            primary_keys: Optional iterable of column names defining the conflict
                target. If omitted, the table's primary keys are used (if any).
        """
        if df.empty:
            return

        # Get table columns
        existing_columns = set(self.list_columns(table_name))
        extra_columns = set(df.columns) - existing_columns
        if extra_columns:
            raise ValueError(
                f"DataFrame contains columns not in table '{table_name}': {extra_columns}"
            )

        # Determine primary keys
        if primary_keys is None:
            primary_keys = self.get_primary_keys(table_name)

        if primary_keys:
            primary_keys = list(primary_keys)
            missing_pks = set(primary_keys) - set(df.columns)
            if missing_pks:
                raise ValueError(f"Primary keys missing from dataframe: {missing_pks}")

        temp_name = f"_temp_df_{uuid.uuid4().hex}"
        table_ident = self._quote_ident(table_name)

        try:
            self.con.register(temp_name, df)

            if primary_keys:
                # Upsert logic
                non_pk_cols = [c for c in df.columns if c not in primary_keys]
                pk_expr = ", ".join(self._quote_ident(c) for c in primary_keys)

                if non_pk_cols:
                    update_expr = ", ".join(
                        f"{self._quote_ident(c)} = EXCLUDED.{self._quote_ident(c)}"
                        for c in non_pk_cols
                    )
                    conflict_clause = f"DO UPDATE SET {update_expr}"
                else:
                    conflict_clause = "DO NOTHING"

                sql = f"""
                    INSERT INTO {table_ident}
                    SELECT * FROM {temp_name}
                    ON CONFLICT ({pk_expr})
                    {conflict_clause}
                """
            else:
                # Table has no primary keys → plain insert
                sql = f"""
                    INSERT INTO {table_ident}
                    SELECT * FROM {temp_name}
                """

            self.con.execute(sql)

        finally:
            self.con.unregister(temp_name)

    # ------------------------------------------------------------------
    # Metadata helpers
    # ------------------------------------------------------------------

    def table_exists(self, table_name: str) -> bool:
        result = self.con.execute(
            """
            SELECT COUNT(*)
            FROM information_schema.tables
            WHERE table_name = ?
            """,
            [table_name],
        ).fetchone()
        assert result is not None
        return bool(result[0])

    def list_tables(self) -> list[str]:
        return [
            row[0]
            for row in self.con.execute(
                """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'main'
                ORDER BY table_name
                """
            ).fetchall()
        ]

    def list_columns(self, table_name: str) -> list[str]:
        return [
            row[0]
            for row in self.con.execute(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name = ?
                ORDER BY ordinal_position
                """,
                [table_name],
            ).fetchall()
        ]

    def get_primary_keys(self, table_name: str) -> list[str]:
        rows = self.con.execute(
            """
            SELECT kcu.column_name
            FROM information_schema.table_constraints AS tc
            JOIN information_schema.key_column_usage AS kcu
            ON tc.constraint_name = kcu.constraint_name
            WHERE tc.table_name = ?
            AND tc.constraint_type = 'PRIMARY KEY'
            ORDER BY kcu.ordinal_position
        """,
            [table_name],
        ).fetchall()
        return [row[0] for row in rows]
