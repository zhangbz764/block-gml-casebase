import unittest
from unittest.mock import patch

import pandas as pd
from shapely import wkb
from shapely.geometry import Point, Polygon

try:
    import multicity_database as mdb
except (ImportError, ModuleNotFoundError):
    mdb = None


class RecordingCursor:
    def __init__(self, connection, rows):
        self.connection = connection
        self.rows = rows
        self.itersize = None

    def execute(self, statement, params=None):
        self.connection.statements.append((" ".join(statement.split()), params))

    def fetchall(self):
        return self.rows

    def __iter__(self):
        return iter(self.rows)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def close(self):
        pass


class RecordingConnection:
    def __init__(self, row_batches):
        self.row_batches = list(row_batches)
        self.statements = []
        self.commits = 0
        self.rollbacks = 0

    def cursor(self, *args, **kwargs):
        rows = self.row_batches[len(self.statements)] if len(self.statements) < len(self.row_batches) else []
        return RecordingCursor(self, rows)

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1


class CitySourceTests(unittest.TestCase):
    def test_city_source_accepts_only_safe_catalog_identifiers(self):
        self.assertIsNotNone(mdb)
        source = mdb.CitySource("berlin", "Berlin", 25833)

        self.assertEqual(source.block_table, "block.berlin_blocks")
        self.assertEqual(source.building_table, "lod2.berlin_buildings_lod2")
        self.assertEqual(
            source.surface_table, "lod2.berlin_building_surfaces_lod2"
        )
        with self.assertRaises(ValueError):
            mdb.CitySource("berlin;drop schema lod2", "bad", 25833)

    def test_discovers_every_configured_lod2_city_from_catalog(self):
        self.assertIsNotNone(mdb)
        connection = RecordingConnection(
            [[("berlin", "Berlin", 25833), ("vienna", "Vienna", 31256)]]
        )

        sources = mdb.discover_lod2_cities(connection)

        self.assertEqual([source.city for source in sources], ["berlin", "vienna"])
        statement, params = connection.statements[0]
        self.assertTrue(statement.upper().startswith("SELECT"))
        self.assertIn("lod2_metric_srid IS NOT NULL", statement)
        self.assertIsNone(params)


class ReadBoundaryTests(unittest.TestCase):
    def test_bad_surface_geometry_is_isolated_to_its_building(self):
        footprint = Polygon([(0, 0), (4, 0), (4, 3), (0, 3), (0, 0)])
        ground = Polygon([(0, 0, 0), (4, 0, 0), (4, 3, 0), (0, 3, 0), (0, 0, 0)])
        rows = [
            ("b1", "block-a", "source-1", 6.0, wkb.dumps(footprint), "GroundSurface", wkb.dumps(ground)),
            ("b1", "block-a", "source-1", 6.0, wkb.dumps(footprint), "RoofSurface", wkb.dumps(Point(1, 1, 5))),
        ]
        connection = RecordingConnection([rows])

        records = list(
            mdb.iter_city_buildings(
                connection,
                mdb.CitySource("berlin", "Berlin", 25833),
                ["block-a"],
            )
        )

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].roof_polygons, tuple())
        self.assertIn("invalid_surface_geometry", records[0].source_quality_codes)

    def test_reads_existing_luxembourg_formal_queries_without_mutation(self):
        self.assertIsNotNone(mdb)
        connection = RecordingConnection(
            [[("q1", "stratified", "slope_q1"), ("q2", "boundary", "steep")]]
        )

        frame = mdb.fetch_luxembourg_formal_queries(connection)

        self.assertEqual(frame["block_id"].tolist(), ["q1", "q2"])
        statement, params = connection.statements[0]
        self.assertTrue(statement.upper().startswith("SELECT"))
        self.assertIn("exp_lod2_roof_lu.formal_queries", statement)
        self.assertIsNone(params)

    def test_stream_retains_building_without_roof_or_ground_surface(self):
        self.assertIsNotNone(mdb)
        footprint = Polygon([(0, 0), (4, 0), (4, 3), (0, 3), (0, 0)])
        roof = Polygon([(0, 0, 5), (4, 0, 5), (4, 3, 6), (0, 3, 6), (0, 0, 5)])
        ground = Polygon([(0, 0, 0), (4, 0, 0), (4, 3, 0), (0, 3, 0), (0, 0, 0)])
        rows = [
            ("b1", "block-a", "source-1", 6.0, wkb.dumps(footprint), "GroundSurface", wkb.dumps(ground)),
            ("b1", "block-a", "source-1", 6.0, wkb.dumps(footprint), "RoofSurface", wkb.dumps(roof)),
            ("b2", "block-b", "source-2", None, wkb.dumps(footprint), None, None),
        ]
        connection = RecordingConnection([rows])
        source = mdb.CitySource("berlin", "Berlin", 25833)

        records = list(
            mdb.iter_city_buildings(connection, source, block_ids=["block-b", "block-a"])
        )

        self.assertEqual([record.building_id for record in records], ["b1", "b2"])
        self.assertEqual(len(records[0].roof_polygons), 1)
        self.assertEqual(len(records[0].ground_polygons), 1)
        self.assertEqual(records[1].roof_polygons, tuple())
        self.assertEqual(records[1].ground_polygons, tuple())
        statement, params = connection.statements[0]
        self.assertIn("LEFT JOIN", statement.upper())
        self.assertEqual(params, (25833, 25833, ["block-a", "block-b"]))


class WriteBoundaryTests(unittest.TestCase):
    @staticmethod
    def _empty_frames():
        return (
            pd.DataFrame(columns=mdb._SAMPLE_COLUMNS),
            pd.DataFrame(columns=mdb._BUILDING_COLUMNS),
            pd.DataFrame(columns=mdb._BLOCK_COLUMNS),
        )

    def test_city_metric_tables_share_one_commit_and_rollback(self):
        manifest, buildings, blocks = self._empty_frames()
        connection = RecordingConnection([])

        with patch.object(mdb, "_replace_partition_in_cursor") as replace:
            mdb.replace_city_metrics(
                connection, "berlin", manifest, buildings, blocks
            )

        self.assertEqual(replace.call_count, 3)
        self.assertEqual(connection.commits, 1)
        self.assertEqual(connection.rollbacks, 0)

        failing = RecordingConnection([])
        with patch.object(
            mdb,
            "_replace_partition_in_cursor",
            side_effect=[None, RuntimeError("insert failed")],
        ):
            with self.assertRaisesRegex(RuntimeError, "insert failed"):
                mdb.replace_city_metrics(
                    failing, "berlin", manifest, buildings, blocks
                )
        self.assertEqual(failing.commits, 0)
        self.assertEqual(failing.rollbacks, 1)

    def test_large_frames_are_converted_to_bounded_insert_batches(self):
        self.assertIsNotNone(mdb)
        frame = pd.DataFrame({"city": ["a"] * 7, "block_id": range(7)})

        batches = list(
            mdb.frame_value_batches(
                frame, ["city", "block_id"], batch_size=3
            )
        )

        self.assertEqual([len(batch) for batch in batches], [3, 3, 1])
        self.assertEqual(batches[0][0], ("a", 0))
        self.assertEqual(batches[-1][-1], ("a", 6))

    def test_schema_and_relation_allowlist_never_target_source_schemas(self):
        self.assertIsNotNone(mdb)
        self.assertEqual(
            mdb.assert_multicity_relation(mdb.BLOCK_TABLE), mdb.BLOCK_TABLE
        )
        with self.assertRaises(ValueError):
            mdb.assert_multicity_relation("lod2.berlin_buildings_lod2")

        connection = RecordingConnection([])
        mdb.ensure_multicity_schema(connection)
        emitted = " ".join(statement for statement, _ in connection.statements)
        self.assertIn(
            f"CREATE SCHEMA IF NOT EXISTS {mdb.EXPERIMENT_SCHEMA}", emitted
        )
        for source_schema in ("block.", "lod1.", "lod2.", "summary."):
            self.assertNotIn(f"CREATE TABLE IF NOT EXISTS {source_schema}", emitted)
            self.assertNotIn(f"DELETE FROM {source_schema}", emitted)
        retrieval_ddl = next(
            statement
            for statement, _ in connection.statements
            if f"CREATE TABLE IF NOT EXISTS {mdb.RETRIEVAL_TABLE}" in statement
        )
        self.assertIn(
            "PRIMARY KEY (query_city, query_id, method, lambda_value, rank)",
            retrieval_ddl,
        )
        self.assertEqual(connection.commits, 1)


if __name__ == "__main__":
    unittest.main()
