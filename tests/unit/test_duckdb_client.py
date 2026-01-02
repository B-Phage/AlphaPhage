import unittest

import duckdb
import pandas as pd
from utils.duckdb_client import DuckDBClient


class TestDuckDBClient(unittest.TestCase):

    def setUp(self):
        # Use in-memory DB for tests
        self.db = DuckDBClient()
        # Create test table
        self.db.con.execute(
            """
            CREATE TABLE items (
                id INTEGER PRIMARY KEY,
                name VARCHAR
            )
        """
        )
        df = pd.DataFrame({"id": [1, 2, 3], "name": ["apple", "banana", "cherry"]})
        self.db.insert_dataframe("items", df, primary_keys=["id"])

    def tearDown(self):
        self.db.close()

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
        # Insert new row
        df_new = pd.DataFrame({"id": [4], "name": ["date"]})
        self.db.insert_dataframe("items", df_new, primary_keys=["id"])

        count = self.db.con.execute("SELECT COUNT(*) FROM items").fetchone()
        self.assertIsNotNone(count)  # runtime check
        assert count is not None  # Pylance type narrowing
        self.assertEqual(count[0], 4)

        # Upsert existing row
        df_update = pd.DataFrame({"id": [2], "name": ["blueberry"]})
        self.db.insert_dataframe("items", df_update, primary_keys=["id"])

        row = self.db.con.execute("SELECT name FROM items WHERE id = 2").fetchone()
        self.assertIsNotNone(row)
        assert row is not None
        self.assertEqual(row[0], "blueberry")

    def test_insert_dataframe_edge_cases(self):
        # 1️⃣ Empty DataFrame -> should do nothing and not raise
        empty_df = pd.DataFrame(columns=["id", "name"])
        try:
            self.db.insert_dataframe("items", empty_df, primary_keys=["id"])
        except Exception as e:
            self.fail(f"insert_dataframe raised an exception on empty DataFrame: {e}")

        # Verify count remains unchanged (still 3 from setUp)
        count = self.db.con.execute("SELECT COUNT(*) FROM items").fetchone()
        self.assertIsNotNone(count)
        assert count is not None
        self.assertEqual(count[0], 3)

        # 2️⃣ Missing primary keys -> should raise ValueError
        df_missing_pk = pd.DataFrame({"name": ["kiwi"]})
        with self.assertRaises(ValueError) as cm:
            self.db.insert_dataframe("items", df_missing_pk, primary_keys=["id"])
        self.assertIn("Primary keys missing from dataframe", str(cm.exception))

        # 3️⃣ No non-PK columns -> conflict_clause = DO NOTHING
        # Insert a DataFrame with all keys primary
        self.db.con.execute(
            """
            CREATE TABLE items_all_pk (
                id INTEGER,
                name VARCHAR,
                PRIMARY KEY (id, name)
            )
        """
        )

        df_only_pk = pd.DataFrame({"id": [6], "name": ["mulbery"]})
        try:
            self.db.insert_dataframe("items_all_pk", df_only_pk)
        except Exception as e:
            self.fail(f"insert_dataframe raised an exception unexpectedly: {e}")

        # Check the row exists
        row = self.db.con.execute("SELECT id FROM items_all_pk WHERE id = 6").fetchone()
        self.assertIsNotNone(row)
        assert row is not None
        self.assertEqual(row[0], 6)

        # Attempt to insert the same PK again -> should DO NOTHING, no exception
        try:
            self.db.insert_dataframe("items_all_pk", df_only_pk)
        except Exception as e:
            self.fail(f"insert_dataframe raised an exception when DO NOTHING expected: {e}")

        # Count should remain the same (no duplicate added)
        count_after = self.db.con.execute("SELECT COUNT(*) FROM items_all_pk").fetchone()
        self.assertIsNotNone(count_after)
        assert count_after is not None
        self.assertEqual(count_after[0], 1)  # 1 inserted

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

        df_only_pk = pd.DataFrame({"id": [6], "name": ["mulbery"]})
        self.db.insert_dataframe("items_no_pk", df_only_pk)
        # Count should be one
        count_after = self.db.con.execute("SELECT COUNT(*) FROM items_no_pk").fetchone()
        self.assertIsNotNone(count_after)
        assert count_after is not None
        self.assertEqual(count_after[0], 1)  # 1 inserted

    def test_execute_raw_sql(self):
        row = self.db.execute("SELECT name FROM items WHERE id = ?", (1,)).fetchone()
        self.assertIsNotNone(row)
        assert row is not None
        self.assertEqual(row[0], "apple")

    def test_run_query_file(self):
        # Create temporary SQL file
        import tempfile

        with tempfile.NamedTemporaryFile("w+", suffix=".sql") as f:
            f.write("SELECT COUNT(*) FROM {table}")
            f.flush()
            rel = self.db.run_query_file(f.name, replacements={"table": "items"})
            count = rel.execute().fetchone()
            self.assertIsNotNone(count)
            assert count is not None
            self.assertEqual(count[0], 3)  # 3 initial

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


if __name__ == "__main__":
    unittest.main()
