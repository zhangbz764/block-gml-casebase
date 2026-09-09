import unittest

import pandas as pd
from shapely import wkb
from shapely.geometry import Polygon

try:
    from visualization import (
        build_comparison_plan,
        fetch_luxembourg_scenes,
    )
except ModuleNotFoundError:
    build_comparison_plan = None
    fetch_luxembourg_scenes = None


class RecordingCursor:
    def __init__(self, connection):
        self.connection = connection
        self.rows = []

    def execute(self, statement, params=None):
        normalized = " ".join(statement.split())
        self.connection.statements.append((normalized, params))
        self.rows = self.connection.row_batches[len(self.connection.statements) - 1]

    def fetchall(self):
        return self.rows

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False


class RecordingConnection:
    def __init__(self, row_batches):
        self.row_batches = row_batches
        self.statements = []

    def cursor(self):
        return RecordingCursor(self)


class ComparisonPlanTests(unittest.TestCase):
    def setUp(self):
        self.retrieval = pd.DataFrame(
            [
                {
                    "query_id": "q",
                    "method": method,
                    "rank": rank,
                    "candidate_id": f"{prefix}{rank}",
                    "d_general": rank / 10,
                    "d_roof": rank / 20,
                }
                for method, prefix in (("M1", "a"), ("M2", "b"))
                for rank in range(1, 6)
            ]
        )

    def test_places_query_then_five_candidates_in_each_method_row(self):
        self.assertIsNotNone(build_comparison_plan)
        result = build_comparison_plan("q", self.retrieval, top_k=5)

        self.assertEqual(len(result), 12)
        self.assertEqual(
            [panel.block_id for panel in result],
            ["q", "a1", "a2", "a3", "a4", "a5", "q", "b1", "b2", "b3", "b4", "b5"],
        )
        self.assertEqual([panel.row for panel in result], [0] * 6 + [1] * 6)
        self.assertEqual([panel.column for panel in result], list(range(6)) * 2)
        self.assertEqual(result[0].title, "Query\nq")
        self.assertIn("rank 1", result[1].title)

    def test_rejects_missing_or_duplicate_ranks(self):
        self.assertIsNotNone(build_comparison_plan)
        broken = self.retrieval[self.retrieval["rank"] != 5].copy()
        with self.assertRaisesRegex(ValueError, "ranks"):
            build_comparison_plan("q", broken, top_k=5)


class SceneFetchTests(unittest.TestCase):
    def test_fetches_only_requested_blocks_with_read_only_selects(self):
        self.assertIsNotNone(fetch_luxembourg_scenes)
        block = Polygon([(0, 0), (10, 0), (10, 8), (0, 8), (0, 0)])
        roof = Polygon([(1, 1, 5), (4, 1, 5), (4, 4, 7), (1, 4, 7), (1, 1, 5)])
        wall = Polygon([(1, 1, 0), (4, 1, 0), (4, 1, 5), (1, 1, 5), (1, 1, 0)])
        connection = RecordingConnection(
            [
                [("q", wkb.dumps(block))],
                [("q", "RoofSurface", wkb.dumps(roof)), ("q", "WallSurface", wkb.dumps(wall))],
            ]
        )

        scenes = fetch_luxembourg_scenes(connection, ["q"], metric_srid=2169)

        self.assertEqual(set(scenes), {"q"})
        self.assertEqual(len(scenes["q"].surfaces), 2)
        self.assertEqual(
            {surface.surface_type for surface in scenes["q"].surfaces},
            {"RoofSurface", "WallSurface"},
        )
        self.assertEqual(len(connection.statements), 2)
        for statement, params in connection.statements:
            self.assertTrue(statement.lstrip().upper().startswith("SELECT"))
            self.assertNotRegex(statement.upper(), r"\b(?:INSERT|UPDATE|DELETE|TRUNCATE|DROP|ALTER)\b")
            self.assertIn(["q"], params)


if __name__ == "__main__":
    unittest.main()
