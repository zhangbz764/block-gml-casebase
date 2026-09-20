from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

from shapely.geometry import Polygon


PHASE2_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PHASE2_DIR))

from graph_encoding import (  # noqa: E402
    build_edges,
    derive_node,
    read_graph_json,
    validate_graph,
    write_graph_json,
)
from run_phase2 import (  # noqa: E402
    SAMPLE_DEFINITIONS,
    assemble_graph,
    freeze_phase1_files,
)


def square(x: float, y: float, size: float = 1.0) -> Polygon:
    return Polygon(
        [(x, y), (x + size, y), (x + size, y + size), (x, y + size)]
    )


def node(node_id: str, x: float, y: float, **values):
    base = {
        "building_id": node_id,
        "representative_point_x_m": x,
        "representative_point_y_m": y,
        "height_m": 10.0,
        "footprint_area_m2": 1.0,
        "orientation_deg": 0.0,
        "geometry_wkt_metric": square(x - 0.5, y - 0.5).wkt,
    }
    base.update(values)
    return base


class GraphEncodingTests(unittest.TestCase):
    def test_sample_plan_contains_twelve_regular_and_three_boundary_samples(self):
        self.assertEqual(15, len(SAMPLE_DEFINITIONS))
        self.assertEqual(12, sum(not item["is_boundary"] for item in SAMPLE_DEFINITIONS))
        self.assertEqual(3, sum(item["is_boundary"] for item in SAMPLE_DEFINITIONS))
        self.assertEqual(15, len({item["block_id"] for item in SAMPLE_DEFINITIONS}))
        contrast_cities = {
            item["city"]
            for item in SAMPLE_DEFINITIONS
            if item["sample_category"] == "形态对照"
        }
        self.assertEqual(3, len(contrast_cities))

    def test_phase1_freeze_manifest_is_sorted_and_hashes_file_content(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "b.txt").write_text("B", encoding="utf-8")
            (root / "a.txt").write_text("A", encoding="utf-8")

            rows = freeze_phase1_files(root)

        self.assertEqual(["a.txt", "b.txt"], [row["path"] for row in rows])
        self.assertEqual(64, len(rows[0]["sha256"]))

    def test_assemble_graph_keeps_metrics_sources_and_local_scale(self):
        definition = SAMPLE_DEFINITIONS[0]
        block = {
            "block_id": definition["block_id"],
            "city": definition["city"],
            "country": "Germany",
            "city_query": "berlin",
            "metric_srid": 25833,
            "storage_srid": 4326,
            "geometry": square(0.0, 0.0, 10.0),
            "area_m2": 100.0,
            "bcr": 0.2,
            "far": None,
            "lod1_building_count": 2,
        }
        buildings = [
            {
                "building_id": "b1",
                "citygml_id": "source-1",
                "height": 8.0,
                "area": 1.0,
                "perimeter": 4.0,
                "geometry": square(2.0, 2.0),
            },
            {
                "building_id": "b2",
                "citygml_id": "source-2",
                "height": 12.0,
                "area": 1.0,
                "perimeter": 4.0,
                "geometry": square(7.0, 7.0),
            },
        ]

        graph = assemble_graph(definition, block, buildings)

        self.assertEqual("0.1", graph["encoding_version"])
        self.assertEqual(2, graph["graph"]["node_count"])
        self.assertEqual(1, graph["graph"]["edge_count"])
        self.assertEqual(10.0, graph["graph"]["coordinate_scale_m"])
        self.assertEqual("lod1.berlin_buildings_lod1", graph["graph"]["source_index"]["building_relation"])
        self.assertTrue(graph["block_attributes"]["missing"]["far"])
        self.assertEqual("two_node_rule", graph["diagnostics"]["edge_construction_status"])

    def test_single_node_is_a_legal_edgeless_graph(self):
        edges, diagnostics = build_edges([node("b1", 0.0, 0.0)], scale_m=10.0)

        self.assertEqual([], edges)
        self.assertEqual("single_node", diagnostics["edge_construction_status"])

    def test_two_nodes_create_one_undirected_edge(self):
        edges, diagnostics = build_edges(
            [node("b1", 0.0, 0.0), node("b2", 3.0, 4.0)], scale_m=10.0
        )

        self.assertEqual(1, len(edges))
        self.assertEqual(["b1", "b2"], [edges[0]["source"], edges[0]["target"]])
        self.assertAlmostEqual(5.0, edges[0]["centroid_distance_m"])
        self.assertAlmostEqual(0.5, edges[0]["centroid_distance_normalized"])
        self.assertEqual("two_node_rule", diagnostics["edge_construction_status"])

    def test_delaunay_triangle_has_three_unique_edges(self):
        edges, diagnostics = build_edges(
            [
                node("b1", 0.0, 0.0),
                node("b2", 1.0, 0.0),
                node("b3", 0.0, 1.0),
            ],
            scale_m=2.0,
        )

        pairs = {(edge["source"], edge["target"]) for edge in edges}
        self.assertEqual({("b1", "b2"), ("b1", "b3"), ("b2", "b3")}, pairs)
        self.assertEqual("delaunay", diagnostics["edge_construction_status"])

    def test_duplicate_points_are_flagged_without_coordinate_jitter(self):
        duplicated = [
            node("b1", 0.0, 0.0),
            node("b2", 0.0, 0.0),
            node("b3", 1.0, 0.0),
        ]

        edges, diagnostics = build_edges(duplicated, scale_m=2.0)

        self.assertEqual([], edges)
        self.assertEqual(
            "degenerate_duplicate_points", diagnostics["edge_construction_status"]
        )
        self.assertEqual([["b1", "b2"]], diagnostics["duplicate_point_groups"])
        self.assertEqual(
            [(0.0, 0.0), (0.0, 0.0), (1.0, 0.0)],
            [
                (item["representative_point_x_m"], item["representative_point_y_m"])
                for item in duplicated
            ],
        )

    def test_node_derivation_preserves_missing_height_and_normalizes_coordinates(self):
        block = square(0.0, 0.0, 10.0)
        record = {
            "building_id": "b1",
            "citygml_id": "source-1",
            "height": None,
            "area": None,
            "perimeter": None,
            "geometry": square(6.0, 4.0, 2.0),
        }

        result = derive_node(
            record,
            block_geometry=block,
            origin_x_m=5.0,
            origin_y_m=5.0,
            scale_m=10.0,
        )

        self.assertIsNone(result["height_m"])
        self.assertTrue(result["missing"]["height_m"])
        self.assertAlmostEqual(4.0, result["footprint_area_m2"])
        self.assertAlmostEqual(8.0, result["footprint_perimeter_m"])
        self.assertAlmostEqual(0.2, result["local_x_normalized"])
        self.assertAlmostEqual(0.0, result["local_y_normalized"])
        self.assertTrue(math.isfinite(result["orientation_sin_2theta"]))
        self.assertTrue(math.isfinite(result["orientation_cos_2theta"]))

    def test_node_flags_footprint_crossing_block_even_when_centroid_is_covered(self):
        block = square(0.0, 0.0, 10.0)
        record = {
            "building_id": "b-crossing",
            "citygml_id": None,
            "height": 5.0,
            "area": 4.0,
            "perimeter": 8.0,
            "geometry": square(8.5, 4.0, 2.0),
        }

        result = derive_node(
            record,
            block_geometry=block,
            origin_x_m=5.0,
            origin_y_m=5.0,
            scale_m=10.0,
        )

        self.assertTrue(result["quality"]["centroid_covered_by_block"])
        self.assertFalse(result["quality"]["footprint_covered_by_block"])

    def test_roundtrip_preserves_nodes_edges_and_missing_values(self):
        graph = {
            "encoding_version": "0.1",
            "graph": {"block_id": "test-block", "node_count": 1},
            "nodes": [{"building_id": "b1", "height_m": None}],
            "edges": [],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "graph.json"
            write_graph_json(graph, path)
            loaded = read_graph_json(path)

        self.assertEqual(graph, loaded)
        self.assertIsNone(loaded["nodes"][0]["height_m"])

    def test_validation_rejects_self_loops_and_duplicate_undirected_edges(self):
        graph = {
            "graph": {"node_count": 2},
            "nodes": [{"building_id": "b1"}, {"building_id": "b2"}],
            "edges": [
                {"source": "b1", "target": "b1", "centroid_distance_m": 0.0},
                {"source": "b1", "target": "b2", "centroid_distance_m": 1.0},
                {"source": "b2", "target": "b1", "centroid_distance_m": 1.0},
            ],
        }

        result = validate_graph(graph, expected_node_count=2)

        self.assertFalse(result["passed"])
        self.assertEqual(1, result["self_loop_count"])
        self.assertEqual(1, result["duplicate_undirected_edge_count"])


if __name__ == "__main__":
    unittest.main()
