from __future__ import annotations

import sys
import unittest
from pathlib import Path

from shapely.geometry import box


PHASE_DIR = Path(__file__).resolve().parents[1]
CH03_DIR = PHASE_DIR.parent
PHASE3_DIR = CH03_DIR / "phase3_encoding_dataset_v03"
sys.path.insert(0, str(PHASE_DIR))
sys.path.insert(0, str(PHASE3_DIR))

from export_phase3 import DEFAULT_TESSELLATION_CONFIG, encode_block_to_v03
from export_phase3_1 import encode_block_to_v031


TEST_SAMPLE = {"sample_id": "p31__TEST", "sample_groups": "unit_test"}
TEST_BLOCK = {
    "block_id": "TEST",
    "city": "Test City",
    "country": "Testland",
    "city_query": "test_city",
    "storage_srid": 3857,
    "metric_srid": 3857,
    "area_m2": 200.0,
    "bcr": 0.2,
    "avg_nn_distance": 6.0,
    "geometry": box(0, 0, 20, 10),
}


def building(building_id, geometry):
    return {
        "building_id": building_id,
        "citygml_id": building_id,
        "height": 8.0,
        "area": geometry.area,
        "perimeter": geometry.length,
        "geometry": geometry,
    }


def structure_signature(record):
    return {
        "nodes": [item["building_id"] for item in record["nodes"]],
        "cells": [item["building_id"] for item in record["cells"]],
        "edges": [item["edge_id"] for item in record["edges"]],
    }


def encode_fixture(buildings):
    return encode_block_to_v031(
        TEST_SAMPLE,
        TEST_BLOCK,
        buildings,
        sample_manifest_sha256="test-manifest",
        tessellation_config=DEFAULT_TESSELLATION_CONFIG,
    )


def encode_fixture_v03(buildings):
    return encode_block_to_v03(
        TEST_SAMPLE,
        TEST_BLOCK,
        buildings,
        sample_manifest_sha256="test-manifest",
        tessellation_config=DEFAULT_TESSELLATION_CONFIG,
    )


class Phase31ExportTests(unittest.TestCase):
    def test_all_collapsed_buildings_exclude_whole_graph(self):
        outcome = encode_fixture([building("sliver", box(2, 2, 2.4, 8))])

        self.assertEqual(outcome.status, "excluded_no_valid_building")
        self.assertIsNone(outcome.record)
        self.assertEqual(len(outcome.excluded_buildings), 1)

    def test_single_remaining_building_is_valid_edgeless_graph(self):
        outcome = encode_fixture(
            [
                building("normal", box(2, 2, 6, 6)),
                building("sliver", box(8, 2, 8.4, 8)),
            ]
        )

        self.assertEqual(outcome.status, "completed")
        self.assertEqual(outcome.record["graph"]["node_count"], 1)
        self.assertEqual(outcome.record["graph"]["cell_count"], 1)
        self.assertEqual(outcome.record["graph"]["cell_failure_count"], 0)
        self.assertEqual(outcome.record["graph"]["edge_count"], 0)
        excluded_id = outcome.excluded_buildings[0]["building_id"]
        self.assertNotIn(excluded_id, {item["building_id"] for item in outcome.record["nodes"]})
        self.assertNotIn(excluded_id, {item["building_id"] for item in outcome.record["cells"]})

    def test_unaffected_two_building_signature_matches_v03(self):
        buildings = [
            building("a", box(2, 2, 6, 6)),
            building("b", box(12, 2, 16, 6)),
        ]

        old = encode_fixture_v03(buildings)
        new = encode_fixture(buildings).record

        self.assertEqual(structure_signature(old), structure_signature(new))


if __name__ == "__main__":
    unittest.main()
