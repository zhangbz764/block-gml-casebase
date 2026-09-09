import unittest

import numpy as np
import pandas as pd
from shapely.geometry import Polygon

from block_metrics import (
    compute_block_metrics,
    compute_building_metrics,
    compute_building_metric_row,
    nearest_neighbour_distances,
    select_final_m1_features,
)
from database import BuildingGeometryRecord


def _building_row(
    building_id,
    footprint,
    slope,
    *,
    usable=True,
    relief=0.2,
    height=10.0,
    roof_area=100.0,
):
    return {
        "building_id": building_id,
        "block_id": "block-1",
        "footprint_area": footprint.area,
        "centroid_x": footprint.centroid.x,
        "centroid_y": footprint.centroid.y,
        "orientation_deg": 0.0,
        "h_g1": height,
        "roof_area_3d": roof_area,
        "roof_projected_area": footprint.area,
        "roof_mean_slope": slope,
        "roof_relief_ratio": relief,
        "roof_surface_count": 1,
        "ground_surface_count": 1,
        "valid_triangle_count": 2,
        "is_usable": usable,
        "quality_codes": "",
    }


class BuildingMetricTests(unittest.TestCase):
    def test_missing_roof_is_a_hard_quality_failure(self):
        footprint = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        ground = Polygon([(0, 0, 0), (10, 0, 0), (10, 10, 0), (0, 10, 0)])
        record = BuildingGeometryRecord(
            "b1", "block-1", None, None, footprint, tuple(), (ground,)
        )

        result = compute_building_metric_row(record)

        self.assertFalse(result["is_usable"])
        self.assertIn("missing_roof", result["quality_codes"])


class BlockMetricTests(unittest.TestCase):
    def setUp(self):
        self.block = Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])
        self.blocks = pd.DataFrame(
            [{"block_id": "block-1", "area_m2": 10000.0,
              "perimeter_m": 400.0, "geometry": self.block}]
        )

    def test_aggregates_lod1_and_roof_metrics(self):
        buildings = pd.DataFrame(
            [
                _building_row(
                    "b1", Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]), 0.0
                ),
                _building_row(
                    "b2", Polygon([(20, 0), (30, 0), (30, 10), (20, 10)]), 45.0
                ),
            ]
        )

        result = compute_block_metrics(self.blocks, buildings).iloc[0]

        self.assertAlmostEqual(result["bcr"], 0.02)
        self.assertAlmostEqual(result["volume_density"], 0.02 * 10.0)
        self.assertEqual(result["building_count"], 2)
        self.assertAlmostEqual(result["roof_mean_slope"], 22.5)
        self.assertAlmostEqual(result["building_roof_slope_sd"], 22.5)
        self.assertTrue(result["is_usable"])

    def test_nearest_neighbour_distances_use_pairwise_metric_distance(self):
        distances = nearest_neighbour_distances(
            np.array([0.0, 3.0, 6.0]),
            np.array([0.0, 4.0, 8.0]),
        )

        np.testing.assert_allclose(distances, [5.0, 5.0, 5.0])

    def test_low_usable_footprint_coverage_excludes_block(self):
        buildings = pd.DataFrame(
            [
                _building_row(
                    "good", Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]), 0.0
                ),
                _building_row(
                    "bad",
                    Polygon([(20, 0), (50, 0), (50, 30), (20, 30)]),
                    np.nan,
                    usable=False,
                    height=np.nan,
                ),
            ]
        )

        result = compute_block_metrics(self.blocks, buildings).iloc[0]

        self.assertAlmostEqual(result["usable_footprint_ratio"], 0.1)
        self.assertFalse(result["is_usable"])
        self.assertIn("coverage_below_80pct", result["quality_codes"])

    def test_empty_building_batch_still_returns_a_quality_failure(self):
        buildings = compute_building_metrics([])

        result = compute_block_metrics(self.blocks, buildings).iloc[0]

        self.assertFalse(result["is_usable"])
        self.assertIn("no_usable_buildings", result["quality_codes"])

    def test_single_building_has_zero_roof_slope_sd(self):
        buildings = pd.DataFrame(
            [_building_row(
                "b1", Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]), 12.0
            )]
        )

        result = compute_block_metrics(self.blocks, buildings).iloc[0]

        self.assertEqual(result["building_roof_slope_sd"], 0.0)

    def test_correlation_pruning_is_deterministic(self):
        frame = pd.DataFrame(
            {
                "log_block_area": [1, 2, 3, 4],
                "compactness": [2, 4, 6, 8],
                "elongation": [4, 1, 3, 2],
            }
        )

        retained, decisions = select_final_m1_features(
            frame,
            candidate_features=["log_block_area", "compactness", "elongation"],
        )

        self.assertEqual(retained, ["log_block_area", "elongation"])
        self.assertEqual(decisions.iloc[0]["dropped_feature"], "compactness")
        self.assertEqual(decisions.iloc[0]["reason"], "same_missingness_drop_later")


if __name__ == "__main__":
    unittest.main()
