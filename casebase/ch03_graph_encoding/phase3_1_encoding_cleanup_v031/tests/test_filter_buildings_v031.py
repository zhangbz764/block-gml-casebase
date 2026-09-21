from __future__ import annotations

import sys
import unittest
from pathlib import Path

from shapely.geometry import box


PHASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PHASE_DIR))

from filter_buildings_v031 import partition_buildings_for_encoding


def building(building_id, geometry):
    return {
        "building_id": building_id,
        "citygml_id": building_id,
        "height": 8.0,
        "area": geometry.area,
        "perimeter": geometry.length,
        "geometry": geometry,
    }


class BuildingPartitionTests(unittest.TestCase):
    def test_collapsed_building_is_excluded_before_node_creation(self):
        buildings = [
            building("normal", box(2, 2, 6, 6)),
            building("sliver", box(8, 2, 8.4, 8)),
        ]

        kept, excluded = partition_buildings_for_encoding(
            buildings, box(0, 0, 20, 10), 0.4
        )

        self.assertEqual([item["building_id"] for item in kept], ["normal"])
        self.assertEqual([item["building_id"] for item in excluded], ["sliver"])
        self.assertEqual(excluded[0]["reason"], "empty_after_shrink")
        self.assertGreater(excluded[0]["footprint_area_m2"], 0)
        self.assertIn("geometry", buildings[1])


if __name__ == "__main__":
    unittest.main()
