from __future__ import annotations

import sys
import unittest
from pathlib import Path

from shapely.geometry import Polygon, box


PHASE_DIR = Path(__file__).resolve().parents[1]
PHASE3_DIR = PHASE_DIR.parent / "phase3_encoding_dataset_v03"
sys.path.insert(0, str(PHASE_DIR))
sys.path.insert(0, str(PHASE3_DIR))

from export_phase3 import DEFAULT_TESSELLATION_CONFIG
from export_phase4b_v032 import encode_block_to_v032
from filter_buildings_v032 import partition_buildings_for_encoding


SAMPLE = {"sample_id": "P4B_TEST", "sample_groups": "phase4b_test"}
BLOCK = {
    "block_id": "TEST", "city": "Test City", "country": "Testland",
    "city_query": "test_city", "storage_srid": 3857, "metric_srid": 3857,
    "area_m2": 400.0, "bcr": 0.2, "avg_nn_distance": 5.0,
    "geometry": box(0, 0, 20, 20),
}


def building(building_id, geometry):
    return {
        "building_id": building_id, "citygml_id": building_id,
        "height": 8.0, "area": geometry.area, "perimeter": geometry.length,
        "geometry": geometry,
    }


class FilterAndExportTests(unittest.TestCase):
    def test_defined_geometry_failures_are_excluded_by_reason(self):
        degenerate = Polygon([(1, 1), (2, 2), (3, 3), (1, 1)])
        sliver = box(5, 5, 5.4, 9)
        kept, excluded = partition_buildings_for_encoding(
            [building("degenerate", degenerate), building("sliver", sliver)],
            BLOCK["geometry"],
            shrink_m=0.4,
        )

        self.assertEqual(kept, [])
        self.assertEqual(
            {item["building_id"]: item["reason"] for item in excluded},
            {
                "degenerate": "invalid_polygonal_area_before_shrink",
                "sliver": "empty_after_shrink",
            },
        )

    def test_one_valid_building_survives_and_graph_counts_are_conserved(self):
        degenerate = Polygon([(1, 1), (2, 2), (3, 3), (1, 1)])
        outcome = encode_block_to_v032(
            SAMPLE,
            BLOCK,
            [building("bad", degenerate), building("good", box(10, 10, 14, 14))],
            sample_manifest_sha256="manifest",
            tessellation_config=DEFAULT_TESSELLATION_CONFIG,
        )

        self.assertEqual(outcome.status, "completed")
        self.assertTrue(outcome.record["validation"]["passed"])
        self.assertEqual(outcome.record["encoding_version"], "0.3.2")
        self.assertEqual(outcome.record["graph"]["node_count"], 1)
        self.assertEqual(outcome.record["graph"]["cell_count"], 1)
        self.assertEqual(outcome.record["graph"]["edge_count"], 0)
        self.assertEqual(outcome.record["graph"]["source_building_count"], 2)
        self.assertEqual(outcome.record["graph"]["encoded_building_count"], 1)
        self.assertEqual(outcome.record["graph"]["excluded_building_count"], 1)
        self.assertNotIn("bad", {item["building_id"] for item in outcome.record["nodes"]})

    def test_all_invalid_buildings_exclude_graph(self):
        degenerate = Polygon([(1, 1), (2, 2), (3, 3), (1, 1)])
        outcome = encode_block_to_v032(
            SAMPLE, BLOCK, [building("bad", degenerate)],
            sample_manifest_sha256="manifest",
            tessellation_config=DEFAULT_TESSELLATION_CONFIG,
        )

        self.assertEqual(outcome.status, "excluded_no_valid_building")
        self.assertIsNone(outcome.record)


if __name__ == "__main__":
    unittest.main()
