import unittest

import database

from database import (
    BLOCK_TABLE,
    BUILDING_TABLE,
    EXPERIMENT_SCHEMA,
    RETRIEVAL_TABLE,
    assert_experiment_relation,
    ensure_experiment_schema,
    iter_luxembourg_buildings,
)


class RecordingCursor:
    def __init__(self, statements):
        self.statements = statements

    def execute(self, statement, params=None):
        self.statements.append((" ".join(statement.split()), params))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def __iter__(self):
        return iter(())

    def close(self):
        pass


class RecordingConnection:
    def __init__(self):
        self.statements = []
        self.commits = 0

    def cursor(self, *args, **kwargs):
        return RecordingCursor(self.statements)

    def commit(self):
        self.commits += 1


class DatabaseSafetyTests(unittest.TestCase):
    def test_relation_allowlist_rejects_source_schema(self):
        self.assertEqual(assert_experiment_relation(BUILDING_TABLE), BUILDING_TABLE)
        with self.assertRaises(ValueError):
            assert_experiment_relation("lod2.luxembourg_buildings_lod2")

    def test_schema_setup_only_creates_experiment_relations(self):
        connection = RecordingConnection()

        ensure_experiment_schema(connection)

        emitted = " ".join(statement for statement, _ in connection.statements)
        self.assertIn(f"CREATE SCHEMA IF NOT EXISTS {EXPERIMENT_SCHEMA}", emitted)
        self.assertIn(f"CREATE TABLE IF NOT EXISTS {BUILDING_TABLE}", emitted)
        self.assertIn(f"CREATE TABLE IF NOT EXISTS {BLOCK_TABLE}", emitted)
        self.assertIn(f"CREATE TABLE IF NOT EXISTS {RETRIEVAL_TABLE}", emitted)
        for source_schema in ("block.", "lod1.", "lod2.", "summary."):
            self.assertNotIn(f"CREATE TABLE IF NOT EXISTS {source_schema}", emitted)
            self.assertNotIn(f"TRUNCATE TABLE {source_schema}", emitted)
        self.assertEqual(connection.commits, 1)

    def test_formal_evaluation_relations_are_isolated_and_created(self):
        self.assertTrue(hasattr(database, "FORMAL_QUERY_TABLE"))
        self.assertTrue(hasattr(database, "FORMAL_EVALUATION_TABLE"))
        self.assertTrue(hasattr(database, "replace_formal_queries"))
        self.assertTrue(hasattr(database, "replace_formal_evaluation_results"))
        formal_query_table = database.FORMAL_QUERY_TABLE
        formal_evaluation_table = database.FORMAL_EVALUATION_TABLE
        self.assertTrue(formal_query_table.startswith(f"{EXPERIMENT_SCHEMA}."))
        self.assertTrue(formal_evaluation_table.startswith(f"{EXPERIMENT_SCHEMA}."))
        self.assertEqual(assert_experiment_relation(formal_query_table), formal_query_table)
        self.assertEqual(
            assert_experiment_relation(formal_evaluation_table),
            formal_evaluation_table,
        )

        connection = RecordingConnection()
        ensure_experiment_schema(connection)
        emitted = " ".join(statement for statement, _ in connection.statements)
        self.assertIn(f"CREATE TABLE IF NOT EXISTS {formal_query_table}", emitted)
        self.assertIn(f"CREATE TABLE IF NOT EXISTS {formal_evaluation_table}", emitted)

    def test_building_stream_can_be_limited_to_sampled_blocks(self):
        connection = RecordingConnection()

        list(iter_luxembourg_buildings(connection, 2169, block_ids=["b2", "b1"]))

        statement, params = connection.statements[0]
        self.assertIn("b.block_id = ANY(%s)", statement)
        self.assertEqual(params, (2169, 2169, ["b1", "b2"]))


if __name__ == "__main__":
    unittest.main()
