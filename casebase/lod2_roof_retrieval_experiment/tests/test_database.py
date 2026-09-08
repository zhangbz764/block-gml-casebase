import unittest

from database import (
    BLOCK_TABLE,
    BUILDING_TABLE,
    EXPERIMENT_SCHEMA,
    RETRIEVAL_TABLE,
    assert_experiment_relation,
    ensure_experiment_schema,
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


if __name__ == "__main__":
    unittest.main()
