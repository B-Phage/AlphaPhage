import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import duckdb
import pandas as pd
from storage_io.duckdb_client import DuckDBClient


class TestDuckDBClient(unittest.TestCase):

    def setUp(self):
        # Use in-memory DB for tests
        self.db = DuckDBClient()
        self.test_dir = Path(tempfile.mkdtemp(prefix="duckdb_test_"))

        # Create test table
        self.db.con.execute(
            """
            CREATE TABLE items (
                id INTEGER PRIMARY KEY,
                name VARCHAR
            )
        """
        )
        # GUARDRAIL: Cast to explicit str to avoid 'object' error
        df = pd.DataFrame({"id": [1, 2, 3], "name": ["apple", "banana", "cherry"]}).astype(
            {"name": "string", "id": "int64"}
        )

        self.db.insert_dataframe("items", df, primary_keys=["id"])

    def tearDown(self):
        if hasattr(self, "db"):
            self.db.close()
        if hasattr(self, "test_dir") and self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_table_exists(self):
        self.assertTrue(self.db.table_exists("items"))
        self.assertFalse(self.db.table_exists("nonexistent"))

    def test_list_tables(self):
        tables = self.db.list_tables()
        self.assertIn("items", tables)

    def test_list_columns(self):
        columns = self.db.list_columns("items")
        self.assertEqual(columns, ["id", "name"])

    def test_insert_dataframe_upsert(self):
        # GUARDRAIL: cast name to str
        df_new = pd.DataFrame({"id": [4], "name": ["date"]}).astype(
            {"id": "int64", "name": "string"}
        )
        self.db.insert_dataframe("items", df_new, primary_keys=["id"])

        count = self.db.con.execute("SELECT COUNT(*) FROM items").fetchone()
        self.assertIsNotNone(count)
        assert count is not None  # Pylance type narrowing
        self.assertEqual(count[0], 4)

        # Upsert existing row
        df_update = pd.DataFrame({"id": [2], "name": ["blueberry"]}).astype(
            {"id": "int64", "name": "string"}
        )
        self.db.insert_dataframe("items", df_update, primary_keys=["id"])

        row = self.db.con.execute("SELECT name FROM items WHERE id = 2").fetchone()
        self.assertIsNotNone(row)
        assert row is not None
        self.assertEqual(row[0], "blueberry")

    def test_insert_dataframe_edge_cases(self):
        # 1️⃣ Empty DataFrame -> should do nothing and not raise
        empty_df = pd.DataFrame(columns=["id", "name"]).astype({"name": "string", "id": int})
        self.db.insert_dataframe("items", empty_df, primary_keys=["id"])
        # Verify count remains unchanged (still 3 from setUp)
        count = self.db.con.execute("SELECT COUNT(*) FROM items").fetchone()
        self.assertIsNotNone(count)
        assert count is not None
        self.assertEqual(count[0], 3)

        # 2️⃣ Missing primary keys
        df_missing_pk = pd.DataFrame({"name": ["kiwi"]}).astype("string")
        with self.assertRaises(ValueError):
            self.db.insert_dataframe("items", df_missing_pk, primary_keys=["id"])

        # 3️⃣ No non-PK columns -> conflict_clause = DO NOTHING
        # Insert a DataFrame with all keys primary
        self.db.con.execute(
            "CREATE TABLE items_all_pk (id INTEGER, name VARCHAR, PRIMARY KEY (id, name))"
        )
        df_only_pk = pd.DataFrame({"id": [6], "name": ["mulberry"]}).astype({"name": "string"})
        self.db.insert_dataframe("items_all_pk", df_only_pk)
        # Check the row exists
        row = self.db.con.execute("SELECT id FROM items_all_pk WHERE id = 6").fetchone()
        self.assertIsNotNone(row)
        assert row is not None
        self.assertEqual(row[0], 6)

        # Attempt to insert the same PK again -> should DO NOTHING, no exception
        self.db.insert_dataframe("items_all_pk", df_only_pk)
        # Count should remain the same (no duplicate added)
        count_after = self.db.con.execute("SELECT COUNT(*) FROM items_all_pk").fetchone()
        self.assertIsNotNone(count_after)
        assert count_after is not None
        self.assertEqual(count_after[0], 1)

    def test_insert_dataframe_edge_cases_extra_col(self):

        extra_col_df = pd.DataFrame({"id": [6], "name": ["mulbery"], "class": ["fruit"]})
        with self.assertRaises(ValueError) as cm:
            self.db.insert_dataframe("items", extra_col_df, primary_keys=["id"])

    def test_insert_dataframe_no_primary_key(self):

        # No Primary Key columns
        self.db.con.execute(
            """
            CREATE TABLE items_no_pk (
                id INTEGER,
                name VARCHAR
            )
        """
        )

        df_only_pk = pd.DataFrame({"id": [6], "name": ["mulbery"]}).astype(
            {"name": "string", "id": "int64"}
        )
        self.db.insert_dataframe("items_no_pk", df_only_pk)
        # Count should be one
        count_after = self.db.con.execute("SELECT COUNT(*) FROM items_no_pk").fetchone()
        self.assertIsNotNone(count_after)
        assert count_after is not None
        self.assertEqual(count_after[0], 1)  # 1 inserted

    def test_insert_dataframe_rejects_object_type(self):
        """Verify strict guardrail: Passing default pandas DF (object) should raise TypeError."""
        bad_df = pd.DataFrame({"id": [10], "name": ["forbidden_object"]})
        with self.assertRaises(TypeError) as cm:
            self.db.insert_dataframe("items", bad_df, primary_keys=["id"])
        self.assertIn("object", str(cm.exception).lower())

    def test_execute_raw_sql(self):
        row = self.db.execute("SELECT name FROM items WHERE id = ?", (1,)).fetchone()
        self.assertIsNotNone(row)
        assert row is not None
        self.assertEqual(row[0], "apple")

    def test_run_query_file(self):
        sql_file = self.test_dir / "test_query.sql"
        sql_file.write_text("SELECT COUNT(*) FROM {table}")

        rel = self.db.run_query_file(str(sql_file), replacements={"table": "items"})
        count = rel.execute().fetchone()
        self.assertIsNotNone(count)
        assert count is not None
        self.assertEqual(count[0], 3)

    def test_transaction_commit(self):
        with self.db.transaction():
            self.db.con.execute("INSERT INTO items (id, name) VALUES (?, ?)", (15, "elderberry"))
        count = self.db.con.execute("SELECT COUNT(*) FROM items").fetchone()
        self.assertIsNotNone(count)
        assert count is not None
        self.assertEqual(count[0], 4)  # 3 initial, justed INSERTed one

    def test_transaction_rollback(self):
        try:
            with self.db.transaction():
                self.db.con.execute("INSERT INTO items (id, name) VALUES (?, ?)", (6, "fig"))
                raise RuntimeError("force rollback")
        except RuntimeError:
            pass

        # Check that insert was rolled back
        row = self.db.con.execute("SELECT name FROM items WHERE id = 6").fetchone()
        self.assertIsNone(row)

    def test_context_manager_closes_connection(self):
        # Enter the context manager
        with DuckDBClient(":memory:") as db:
            # __enter__ should return the client itself
            self.assertIsInstance(db, DuckDBClient)

            # Connection should be usable inside the context
            result = db.con.execute("SELECT 1").fetchone()
            self.assertIsNotNone(result)  # runtime check
            assert result is not None  # type narrowing for Pylance
            self.assertEqual(result[0], 1)

        # After exiting the context, the connection should be closed with __exit__
        with self.assertRaises(duckdb.Error):
            db.con.execute("SELECT 1")

    def test_get_table(self):
        rel = self.db.table("items")
        # No query executed yet

        rel = rel.filter("id > 2").project("id, name")  # should return id=3, name=cherry
        # Still no query executed

        df = rel.df()
        # NOW the query runs
        self.assertEqual(df.shape, (1, 2))

    def test_quote_ident_valid(self):
        # Normal identifier is quoted correctly
        result = DuckDBClient._quote_ident("my_table")
        self.assertEqual(result, '"my_table"')

    def test_quote_ident_injection(self):
        # Identifier with double quote should raise ValueError
        with self.assertRaises(ValueError) as cm:
            DuckDBClient._quote_ident('bad"table')
        self.assertIn("Invalid identifier", str(cm.exception))

    def test_init_memory_db(self):
        client = DuckDBClient()
        self.assertEqual(client.db_path, ":memory:")
        self.assertTrue(hasattr(client, "con"))
        self.assertIsInstance(client.con, duckdb.DuckDBPyConnection)

    def test_init_with_config(self):
        # Use a valid DuckDB config option
        config = {"threads": "1"}  # valid
        client = DuckDBClient(db_path=":memory:", config=config)
        self.assertEqual(client.db_path, ":memory:")
        self.assertTrue(hasattr(client, "con"))
        self.assertIsInstance(client.con, duckdb.DuckDBPyConnection)

    @patch("storage_io.duckdb_client.time.sleep", return_value=None)
    @patch("storage_io.duckdb_client.duckdb.connect")
    def test_retry_logic_success_after_lock(self, connect_mock, sleep_mock):
        # Fail twice with lock, then succeed
        connect_mock.side_effect = [
            duckdb.IOException("database is locked"),
            duckdb.IOException("lock detected"),
            MagicMock(),  # succeeds
        ]
        client = DuckDBClient(db_path=":memory:")
        self.assertTrue(hasattr(client, "con"))
        self.assertEqual(connect_mock.call_count, 3)
        self.assertEqual(sleep_mock.call_count, 2)

    @patch("storage_io.duckdb_client.time.sleep", return_value=None)
    @patch("storage_io.duckdb_client.duckdb.connect")
    def test_retry_logic_raises_after_max_attempts(self, connect_mock, sleep_mock):
        # Always raise lock error
        connect_mock.side_effect = duckdb.IOException("database is locked")
        with self.assertRaises(duckdb.IOException) as cm:
            DuckDBClient(db_path=":memory:")
        self.assertIn("lock", str(cm.exception).lower())
        self.assertEqual(connect_mock.call_count, 10)
        self.assertEqual(sleep_mock.call_count, 9)  # last attempt doesn't sleep

    @patch("storage_io.duckdb_client.time.sleep", return_value=None)
    @patch("storage_io.duckdb_client.duckdb.connect")
    def test_retry_logic_raises_other_exception_immediately(self, connect_mock, sleep_mock):
        connect_mock.side_effect = duckdb.IOException("some other IO error")
        with self.assertRaises(duckdb.IOException) as cm:
            DuckDBClient(db_path=":memory:")
        self.assertIn("some other IO error", str(cm.exception))
        self.assertEqual(connect_mock.call_count, 1)
        self.assertEqual(sleep_mock.call_count, 0)

    def test_init_creates_parent_dir_for_file_db(self):
        # Create a temporary directory and a nested path
        temp_dir = self.test_dir / "nested_dir"
        db_file = temp_dir / "test.db"

        # Ensure the parent directory doesn't exist yet
        self.assertFalse(temp_dir.exists())

        # Initialize DuckDBClient with a file-based path
        client = DuckDBClient(db_path=str(db_file))

        # Check that the parent directory was created
        self.assertTrue(temp_dir.exists())
        self.assertTrue(client.db_path == str(db_file))
        self.assertTrue(hasattr(client, "con"))
        self.assertIsInstance(client.con, duckdb.DuckDBPyConnection)

        # Cleanup: close connection and remove file + directory
        client.close()
        if db_file.exists():
            db_file.unlink()
        if temp_dir.exists():
            temp_dir.rmdir()

    def test_head_returns_correct_rows(self):
        """Verify head returns the requested number of rows as a DataFrame."""
        # Table has: {1: 'apple', 2: 'banana', 3: 'cherry'}
        df_head = self.db.head("items", n=2)

        self.assertIsInstance(df_head, pd.DataFrame)
        self.assertEqual(len(df_head), 2)

        # Verify all returned names are actually in our known set
        valid_names = {"apple", "banana", "cherry"}
        returned_names = set(df_head["name"])

        self.assertTrue(returned_names.issubset(valid_names))
        self.assertEqual(len(returned_names), 2)

    def test_head_default_n(self):
        """Verify head defaults to 5 rows (or total table size if smaller)."""
        # Table has 3 rows, default n is 5
        df_head = self.db.head("items")

        self.assertEqual(len(df_head), 3)
        self.assertIsInstance(df_head, pd.DataFrame)

    def test_head_raises_error_for_missing_table(self):
        """Verify head raises ValueError if the table is missing."""
        with self.assertRaises(ValueError) as cm:
            self.db.head("ghost_table")

        self.assertIn("does not exist", str(cm.exception))

    def test_head_with_quoted_identifiers(self):
        """Verify head handles table names that require quoting."""
        # Create a table with a space in the name
        complex_name = "complex items"
        self.db.con.execute(f'CREATE TABLE "{complex_name}" (val INTEGER)')
        self.db.con.execute(f'INSERT INTO "{complex_name}" VALUES (1), (2), (3)')

        df_head = self.db.head(complex_name, n=1)
        self.assertEqual(len(df_head), 1)
        self.assertEqual(df_head.iloc[0]["val"], 1)

    def test_get_table_hash_raises_error_for_missing_table(self):
        """Verify get_table_hash raises ValueError if the table is missing."""
        with self.assertRaises(ValueError) as cm:
            self.db.get_table_hash("non_existent_table")

        self.assertIn("does not exist", str(cm.exception))

    def test_get_table_hash_empty_table(self):
        """Verify get_table_hash returns 'EMPTY_TABLE' for a table with no rows."""
        self.db.con.execute("CREATE TABLE empty_table (id INTEGER, val VARCHAR)")

        result = self.db.get_table_hash("empty_table")
        self.assertEqual(result, "EMPTY_TABLE")

    def test_get_table_hash_determinism(self):
        """Verify the hash is order-independent and content-specific."""
        self.db.con.execute("CREATE TABLE hash_test (id INTEGER, val VARCHAR)")

        # Insert data in one order
        self.db.con.execute("INSERT INTO hash_test VALUES (1, 'A'), (2, 'B')")
        hash_1 = self.db.get_table_hash("hash_test")

        # Create identical table but insert in different order
        self.db.con.execute("CREATE TABLE hash_test_2 (id INTEGER, val VARCHAR)")
        self.db.con.execute("INSERT INTO hash_test_2 VALUES (2, 'B'), (1, 'A')")
        hash_2 = self.db.get_table_hash("hash_test_2")

        # Hashes must match because bit_xor is commutative
        self.assertEqual(hash_1, hash_2)
        self.assertNotEqual(hash_1, "EMPTY_TABLE")

    def test_get_table_hash_handles_nulls(self):
        """Verify hash handles NULL values via COALESCE logic."""
        self.db.con.execute("CREATE TABLE null_test (id INTEGER, val VARCHAR)")
        self.db.con.execute("INSERT INTO null_test (id) VALUES (1)")  # val is NULL

        try:
            hash_val = self.db.get_table_hash("null_test")
            self.assertIsInstance(hash_val, str)
            self.assertNotEqual(hash_val, "EMPTY_TABLE")
        except Exception as e:
            self.fail(f"get_table_hash failed on NULL values: {e}")


if __name__ == "__main__":
    unittest.main()
