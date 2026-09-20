from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

from shapely import wkt
from shapely.geometry import Polygon, box


PHASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PHASE_DIR))

from tessellation_encoding import (  # noqa: E402
    TessellationConfig,
    build_cell_adjacency,
    build_morphological_cells,
    read_json,
    validate_tessellation_graph,
    write_json,
)
from run_phase2_5 import corrected_graph_metadata  # noqa: E402


def building(building_id: str, geometry: Polygon, **values):
    record = {
        "building_id": building_id,
        "geometry_wkt_metric": geometry.wkt,
        "height_m": 10.0,
        "footprint_area_m2": geometry.area,
        "orientation_deg": 0.0,
    }
    record.update(values)
    return record


def cell(building_id: str, geometry: Polygon):
    return {
        "building_id": building_id,
        "geometry_wkt_metric": geometry.wkt,
        "area_m2": geometry.area,
        "perimeter_m": geometry.length,
    }


class MorphologicalCellTests(unittest.TestCase):
    def setUp(self):
        self.config = TessellationConfig(shrink_m=0.4, segment_m=0.5)

    def test_two_rectangles_generate_one_traceable_cell_each(self):
        block = box(0, 0, 10, 6)
        buildings = [
            building("b1", box(1, 2, 3, 4)),
            building("b2", box(7, 2, 9, 4)),
        ]

        cells, failures, diagnostics = build_morphological_cells(
            block, buildings, self.config
        )

        self.assertEqual([], failures)
        self.assertEqual({"b1", "b2"}, {cell["building_id"] for cell in cells})
        self.assertEqual(2, diagnostics["success_cell_count"])
        self.assertTrue(all(wkt.loads(cell["geometry_wkt_metric"]).is_valid for cell in cells))

    def test_cells_stay_inside_block_and_do_not_overlap_in_area(self):
        block = box(0, 0, 10, 6)
        buildings = [
            building("b1", box(1, 2, 3, 4)),
            building("b2", box(7, 2, 9, 4)),
        ]

        cells, _, diagnostics = build_morphological_cells(block, buildings, self.config)
        geometries = [wkt.loads(cell["geometry_wkt_metric"]) for cell in cells]

        self.assertTrue(all(block.covers(geometry) for geometry in geometries))
        self.assertLessEqual(diagnostics["cell_overlap_area_m2"], 1e-8)
        self.assertLessEqual(diagnostics["cell_outside_area_m2"], 1e-8)
        self.assertLessEqual(diagnostics["uncovered_area_m2"], 1e-8)

    def test_single_building_cell_covers_entire_block(self):
        block = box(0, 0, 10, 6)

        cells, failures, diagnostics = build_morphological_cells(
            block, [building("only", box(4, 2, 6, 4))], self.config
        )

        self.assertEqual([], failures)
        self.assertEqual(1, len(cells))
        self.assertAlmostEqual(block.area, wkt.loads(cells[0]["geometry_wkt_metric"]).area)
        self.assertEqual("single_building_rule", diagnostics["construction_status"])

    def test_same_centroid_different_outlines_do_not_require_jitter(self):
        block = box(0, 0, 10, 10)
        buildings = [
            building("horizontal", box(2, 4, 8, 6)),
            building("vertical", box(4, 2, 6, 8)),
        ]

        cells, failures, diagnostics = build_morphological_cells(
            block, buildings, self.config
        )

        self.assertEqual([], failures)
        self.assertEqual(2, len(cells))
        self.assertNotIn("coordinate_jitter", diagnostics["operations"])

    def test_empty_after_shrink_is_reported_with_building_id(self):
        block = box(0, 0, 10, 6)
        buildings = [
            building("too-small", box(1, 1, 1.5, 1.5)),
            building("normal", box(6, 2, 8, 4)),
        ]

        cells, failures, diagnostics = build_morphological_cells(
            block, buildings, self.config
        )

        self.assertEqual(["normal"], [cell["building_id"] for cell in cells])
        self.assertEqual("too-small", failures[0]["building_id"])
        self.assertEqual("empty_after_shrink", failures[0]["failure_type"])
        self.assertTrue(failures[0]["message"])
        self.assertEqual(1, diagnostics["failure_count"])

    def test_cell_json_roundtrip_preserves_id_and_valid_geometry(self):
        payload = {
            "cells": [
                {
                    "building_id": "b1",
                    "geometry_wkt_metric": box(0, 0, 5, 5).wkt,
                }
            ]
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "cells.json"
            write_json(payload, path)
            loaded = read_json(path)

        self.assertEqual("b1", loaded["cells"][0]["building_id"])
        self.assertTrue(wkt.loads(loaded["cells"][0]["geometry_wkt_metric"]).is_valid)

    def test_real_outer_hull_sites_still_form_a_partition(self):
        source = (
            PHASE_DIR.parent
            / "phase2_encoding_v01"
            / "results"
            / "graphs"
            / "trace_tokyo_large.graph.json"
        )
        graph = read_json(source)
        block = wkt.loads(graph["graph"]["block_geometry_wkt_metric"])

        cells, failures, diagnostics = build_morphological_cells(
            block, graph["nodes"], self.config
        )

        self.assertEqual([], failures)
        self.assertEqual(len(graph["nodes"]), len(cells))
        self.assertLessEqual(diagnostics["cell_overlap_area_m2"], 1e-5)
        self.assertLessEqual(diagnostics["uncovered_area_m2"], 1e-5)


class CellAdjacencyTests(unittest.TestCase):
    def setUp(self):
        self.config = TessellationConfig()
        self.block = box(0, 0, 2, 1)
        self.cells = [cell("b1", box(0, 0, 1, 1)), cell("b2", box(1, 0, 2, 1))]
        self.buildings = [
            building("b1", box(0.1, 0.2, 0.9, 0.8), orientation_deg=0.0),
            building("b2", box(1.1, 0.2, 1.9, 0.8), orientation_deg=0.0),
        ]

    def test_point_contact_does_not_create_edge(self):
        cells = [cell("b1", box(0, 0, 1, 1)), cell("b2", box(1, 1, 2, 2))]
        buildings = [building("b1", box(0, 0, 1, 1)), building("b2", box(1, 1, 2, 2))]

        edges, diagnostics = build_cell_adjacency(
            cells, buildings, box(0, 0, 2, 2).area, self.config
        )

        self.assertEqual([], edges)
        self.assertEqual(0, diagnostics["edge_count"])

    def test_shared_segment_creates_one_stable_undirected_edge(self):
        edges, _ = build_cell_adjacency(
            self.cells, self.buildings, self.block.area, self.config
        )

        self.assertEqual(1, len(edges))
        edge = edges[0]
        self.assertEqual("celladj|b1|b2", edge["edge_id"])
        self.assertEqual(("b1", "b2"), (edge["source"], edge["target"]))
        self.assertAlmostEqual(1.0, edge["shared_cell_boundary_m"])
        self.assertGreater(edge["shared_boundary_ratio_min"], 0)
        self.assertGreater(edge["shared_boundary_ratio_mean"], 0)

    def test_orientation_formula_parallel_and_orthogonal(self):
        parallel, _ = build_cell_adjacency(
            self.cells, self.buildings, self.block.area, self.config
        )
        orthogonal_buildings = [
            self.buildings[0],
            building("b2", box(1.1, 0.2, 1.9, 0.8), orientation_deg=90.0),
        ]
        orthogonal, _ = build_cell_adjacency(
            self.cells, orthogonal_buildings, self.block.area, self.config
        )

        self.assertAlmostEqual(0.0, parallel[0]["orientation_diff_deg"])
        self.assertAlmostEqual(1.0, parallel[0]["orientation_similarity"])
        self.assertAlmostEqual(90.0, orthogonal[0]["orientation_diff_deg"])
        self.assertAlmostEqual(0.0, orthogonal[0]["orientation_similarity"])

    def test_missing_height_stays_null_with_reason(self):
        buildings = [self.buildings[0], dict(self.buildings[1], height_m=None)]
        edges, _ = build_cell_adjacency(self.cells, buildings, self.block.area, self.config)
        edge = edges[0]

        for field in ("height_diff_m", "log_height_ratio_abs", "volume_ratio"):
            self.assertIsNone(edge[field])
            self.assertTrue(edge["missing"][field])
            self.assertEqual("missing_or_nonpositive_height", edge["missing_reasons"][field])

    def test_endpoint_input_order_does_not_change_edge(self):
        forward, _ = build_cell_adjacency(
            self.cells, self.buildings, self.block.area, self.config
        )
        reversed_input, _ = build_cell_adjacency(
            list(reversed(self.cells)), list(reversed(self.buildings)), self.block.area, self.config
        )

        self.assertEqual(forward, reversed_input)

    def test_ranges_validation_and_json_roundtrip(self):
        edges, _ = build_cell_adjacency(
            self.cells, self.buildings, self.block.area, self.config
        )
        edge = edges[0]
        for field in (
            "shared_cell_boundary_m",
            "footprint_gap_m",
            "footprint_gap_block_norm",
            "centroid_distance_m",
            "centroid_distance_block_norm",
            "long_side_offset_norm",
            "log_area_ratio_abs",
        ):
            self.assertGreaterEqual(edge[field], 0.0)
        self.assertGreaterEqual(edge["orientation_diff_deg"], 0.0)
        self.assertLessEqual(edge["orientation_diff_deg"], 90.0)
        self.assertGreaterEqual(edge["orientation_similarity"], 0.0)
        self.assertLessEqual(edge["orientation_similarity"], 1.0)
        validation = validate_tessellation_graph(
            {"nodes": self.buildings, "cells": self.cells, "edges": edges}
        )
        self.assertTrue(validation["passed"])
        self.assertEqual([], validation["issues"])

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "graph.json"
            write_json({"nodes": self.buildings, "cells": self.cells, "edges": edges}, path)
            loaded = read_json(path)
        self.assertEqual(edges, loaded["edges"])


class OutputAssemblyTests(unittest.TestCase):
    def test_v02_metadata_uses_actual_arrays_not_v01_edge_count(self):
        source_graph = {
            "block_id": "block-1",
            "node_count": 2,
            "edge_count": 1,
        }
        nodes = [{"building_id": "b1"}, {"building_id": "b2"}]
        cells = [{"building_id": "b1"}, {"building_id": "b2"}]
        failures = []
        v02_edges = []

        metadata = corrected_graph_metadata(
            source_graph, nodes, cells, failures, v02_edges
        )

        self.assertEqual(2, metadata["node_count"])
        self.assertEqual(2, metadata["cell_count"])
        self.assertEqual(0, metadata["cell_failure_count"])
        self.assertEqual(0, metadata["edge_count"])
        self.assertEqual("block-1", metadata["block_id"])


if __name__ == "__main__":
    unittest.main()
