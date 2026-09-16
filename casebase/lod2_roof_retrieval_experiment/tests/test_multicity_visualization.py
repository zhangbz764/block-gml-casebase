import unittest
from types import SimpleNamespace

import pandas as pd
import matplotlib.pyplot as plt
from shapely import wkb
from shapely.geometry import MultiPolygon, Polygon

try:
    import multicity_visualization as mviz
except (ImportError, ModuleNotFoundError):
    mviz = None


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


def retrieval_fixture():
    return pd.DataFrame(
        [
            {
                "query_id": "q1",
                "query_city": "luxembourg",
                "method": method,
                "rank": rank,
                "candidate_id": candidate_id,
                "candidate_city": candidate_city,
                "d_general": d_general,
                "d_roof": d_roof,
                "query_category": "typical",
                "query_category_label": "Typical",
            }
            for method, rank, candidate_id, candidate_city, d_general, d_roof in [
                ("M1", 1, "a", "berlin", 0.10, 1.00),
                ("M1", 2, "b", "vienna", 0.30, 0.60),
                ("M2", 1, "b", "vienna", 0.35, 0.20),
                ("M2", 2, "c", "amsterdam", 0.45, 0.10),
            ]
        ]
    )


class ComparisonPlanTests(unittest.TestCase):
    def test_plan_keeps_city_with_each_scene_and_places_both_method_rows(self):
        self.assertIsNotNone(mviz)

        panels = mviz.build_multicity_comparison_plan(
            "q1", retrieval_fixture(), top_k=2
        )

        self.assertEqual(
            [(panel.city, panel.block_id) for panel in panels],
            [
                ("luxembourg", "q1"),
                ("berlin", "a"),
                ("vienna", "b"),
                ("vienna", "b"),
                ("amsterdam", "c"),
            ],
        )
        self.assertEqual([panel.row for panel in panels], [-1, 0, 0, 1, 1])
        self.assertIn("Typical", panels[0].title)
        self.assertIn("Berlin", panels[1].title)
        self.assertIn("G 0.10", panels[1].title)

    def test_plan_rejects_incomplete_rank_sequence(self):
        broken = retrieval_fixture().query("not (method == 'M2' and rank == 2)")
        with self.assertRaisesRegex(ValueError, "ranks"):
            mviz.build_multicity_comparison_plan("q1", broken, top_k=2)


class QueryDiagnosticTests(unittest.TestCase):
    def test_gain_cost_uses_positive_values_for_roof_gain_and_general_cost(self):
        result = mviz.build_query_gain_cost(retrieval_fixture(), top_k=2)

        self.assertEqual(result["query_id"].tolist(), ["q1"])
        self.assertAlmostEqual(result.loc[0, "roof_gain"], 0.65)
        self.assertAlmostEqual(result.loc[0, "general_cost"], 0.20)
        self.assertEqual(result.loc[0, "top_k_overlap"], 1)
        self.assertEqual(result.loc[0, "replacement_count"], 1)
        self.assertEqual(result.loc[0, "query_category"], "typical")
        self.assertEqual(result.loc[0, "query_category_label"], "Typical")

    def test_numerical_noise_is_reported_as_no_change(self):
        frame = retrieval_fixture()
        frame.loc[frame["method"] == "M1", "d_roof"] = [0.2, 0.4]
        frame.loc[frame["method"] == "M2", "d_roof"] = [0.2, 0.4000000000001]

        result = mviz.build_query_gain_cost(frame, top_k=2)

        self.assertEqual(result.loc[0, "roof_gain"], 0.0)

    def test_query_diagnostics_preserve_the_frozen_query_order(self):
        q1 = retrieval_fixture()
        q2 = q1.assign(query_id="q2")
        ordered = pd.concat([q2, q1], ignore_index=True)

        result = mviz.build_query_gain_cost(ordered, top_k=2)

        self.assertEqual(result["query_id"].tolist(), ["q2", "q1"])

    def test_category_example_is_near_the_category_median_not_the_best_extreme(self):
        diagnostics = pd.DataFrame(
            {
                "query_id": ["low-a", "low-b", "low-c", "high-a", "high-b", "high-c"],
                "query_category": ["low"] * 3 + ["high"] * 3,
                "query_category_label": ["Low"] * 3 + ["High"] * 3,
                "roof_gain": [0.0, 1.0, 10.0, 2.0, 3.0, 20.0],
                "general_cost": [0.0, 1.0, 10.0, 2.0, 3.0, 20.0],
                "replacement_count": [0, 2, 5, 1, 3, 5],
            }
        )

        result = mviz.select_category_examples(diagnostics, per_category=1)

        self.assertEqual(result["query_id"].tolist(), ["low-b", "high-b"])
        self.assertEqual(set(result["example_role"]), {"category_representative"})

    def test_gain_cost_figure_uses_category_labels_when_available(self):
        diagnostics = pd.DataFrame(
            {
                "query_id": ["q1", "q2"],
                "query_category": ["near_flat", "steep"],
                "query_category_label": ["Near-flat", "Steep"],
                "roof_gain": [0.2, 0.5],
                "general_cost": [0.1, 0.2],
                "replacement_count": [2, 4],
            }
        )

        figure = mviz.make_query_gain_cost_figure(diagnostics)
        labels = figure.axes[0].get_legend_handles_labels()[1]
        plt.close(figure)

        self.assertEqual(labels, ["Near-flat", "Steep"])

    def test_effect_ranking_rewards_roof_gain_and_form_preservation_equally(self):
        diagnostics = pd.DataFrame(
            {
                "query_id": ["middle", "best", "worst"],
                "roof_gain": [5.0, 10.0, 1.0],
                "general_cost": [2.0, 1.0, 3.0],
            }
        )

        ranked = mviz.rank_query_effects(diagnostics)

        self.assertEqual(ranked["query_id"].tolist(), ["best", "middle", "worst"])
        self.assertEqual(ranked["effect_rank"].tolist(), [1, 2, 3])
        self.assertEqual(ranked["effect_balance_score"].tolist(), [1.0, 0.5, 0.0])


class SceneFetchTests(unittest.TestCase):
    def test_fetches_city_geometry_read_only_and_expands_multipolygons(self):
        block_a = Polygon([(0, 0), (10, 0), (10, 8), (0, 8), (0, 0)])
        block_b = Polygon([(20, 0), (28, 0), (28, 8), (20, 8), (20, 0)])
        footprint_parts = MultiPolygon(
            [
                Polygon([(1, 1), (3, 1), (3, 3), (1, 3), (1, 1)]),
                Polygon([(4, 1), (6, 1), (6, 3), (4, 3), (4, 1)]),
            ]
        )
        roof_parts = MultiPolygon(
            [
                Polygon([(1, 1, 5), (3, 1, 5), (3, 3, 6), (1, 3, 6), (1, 1, 5)]),
                Polygon([(4, 1, 5), (6, 1, 5), (6, 3, 7), (4, 3, 7), (4, 1, 5)]),
            ]
        )
        connection = RecordingConnection(
            [
                [("a", wkb.dumps(block_a)), ("b", wkb.dumps(block_b))],
                [("a", "building-1", wkb.dumps(footprint_parts))],
                [("a", "RoofSurface", wkb.dumps(roof_parts))],
            ]
        )
        source = SimpleNamespace(
            city="berlin",
            display_name="Berlin",
            metric_srid=25833,
            block_table="block.berlin_blocks",
            building_table="lod2.berlin_buildings_lod2",
            surface_table="lod2.berlin_building_surfaces_lod2",
        )

        scenes = mviz.fetch_multicity_scenes(
            connection,
            {"berlin": source},
            [("berlin", "b"), ("berlin", "a")],
        )

        self.assertEqual(set(scenes), {("berlin", "a"), ("berlin", "b")})
        self.assertEqual(len(scenes[("berlin", "a")].footprints), 2)
        self.assertEqual(len(scenes[("berlin", "a")].surfaces), 2)
        self.assertEqual(len(scenes[("berlin", "b")].footprints), 0)
        self.assertEqual(len(connection.statements), 3)
        for statement, params in connection.statements:
            self.assertTrue(statement.upper().startswith("SELECT"))
            self.assertNotRegex(
                statement.upper(), r"\b(?:INSERT|UPDATE|DELETE|DROP|ALTER)\b"
            )
            self.assertIn(["a", "b"], params)


if __name__ == "__main__":
    unittest.main()
