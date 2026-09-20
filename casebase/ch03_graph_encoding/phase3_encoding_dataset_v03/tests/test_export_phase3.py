from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

from shapely.geometry import box


PHASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PHASE_DIR))

from export_phase3 import (
    apply_weak_edge_flags,
    encode_block_to_v03,
    read_graph,
    validate_v03_record,
    write_graph,
)


class Phase3ExportTests(unittest.TestCase):
    def test_weak_flags_are_added_without_removing_or_reordering_edges(self):
        edges = [
            {
                "edge_id": f"edge-{index}",
                "footprint_gap_block_norm": 10.0 if index == 9 else 0.0,
                "shared_boundary_ratio_min": 0.001 if index == 9 else 0.1 + index,
            }
            for index in range(10)
        ]

        flagged, summary = apply_weak_edge_flags(edges)

        self.assertEqual([edge["edge_id"] for edge in flagged], [f"edge-{index}" for index in range(10)])
        self.assertEqual(summary["long_distance_candidate_count"], 1)
        self.assertEqual(summary["low_shared_candidate_count"], 1)
        self.assertEqual(summary["combined_weak_candidate_count"], 1)
        self.assertTrue(flagged[-1]["combined_weak_candidate"])

    def test_validation_reports_metadata_count_mismatch(self):
        record = {
            "graph": {"node_count": 2, "cell_count": 0, "cell_failure_count": 0, "edge_count": 0},
            "nodes": [{"building_id": "a", "missing": {}, "missing_reasons": {}}],
            "cells": [],
            "cell_failures": [],
            "edges": [],
            "weak_edge_summary": {
                "long_distance_candidate_count": 0,
                "low_shared_candidate_count": 0,
                "combined_weak_candidate_count": 0,
            },
        }

        validation = validate_v03_record(record)

        self.assertFalse(validation["passed"])
        self.assertIn("node_count_mismatch", validation["issue_types"])

    def test_two_building_record_encodes_and_roundtrips_as_gzip_json(self):
        block = {
            "block_id": "TEST_0001",
            "city": "Test City",
            "country": "Testland",
            "city_query": "test_city",
            "storage_srid": 4326,
            "metric_srid": 3857,
            "area_m2": 200.0,
            "bcr": 0.2,
            "avg_nn_distance": 6.0,
            "geometry": box(0, 0, 20, 10),
        }
        buildings = [
            {"building_id": "a", "citygml_id": "a", "height": None, "area": 16.0, "perimeter": 16.0, "geometry": box(2, 3, 6, 7)},
            {"building_id": "b", "citygml_id": "b", "height": 8.0, "area": 16.0, "perimeter": 16.0, "geometry": box(14, 3, 18, 7)},
        ]

        record = encode_block_to_v03(
            {"sample_id": "p3__TEST_0001", "sample_groups": "core_audit"},
            block,
            buildings,
            sample_manifest_sha256="abc123",
        )
        validation = validate_v03_record(record)

        self.assertTrue(validation["passed"], validation["issues"])
        self.assertEqual(record["graph"]["node_count"], 2)
        self.assertEqual(record["graph"]["cell_count"], 2)
        self.assertIsNone(record["nodes"][0]["height_m"])
        self.assertEqual(record["nodes"][0]["missing_reasons"]["height_m"], "source_value_null")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "graph.json.gz"
            write_graph(record, path)
            reloaded = read_graph(path)
        self.assertEqual(
            [edge["edge_id"] for edge in record["edges"]],
            [edge["edge_id"] for edge in reloaded["edges"]],
        )


if __name__ == "__main__":
    unittest.main()
