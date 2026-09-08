import math
import unittest

from shapely.geometry import Polygon

from roof_geometry import g1_height, measure_polygon_3d, roof_relief_ratio


class RoofGeometryTests(unittest.TestCase):
    def test_flat_roof_preserves_area_height_and_zero_slope(self):
        roof = Polygon(
            [(0, 0, 10), (10, 0, 10), (10, 10, 10), (0, 10, 10)]
        )

        result = measure_polygon_3d(roof)

        self.assertAlmostEqual(result.projected_area, 100.0, places=6)
        self.assertAlmostEqual(result.area_3d, 100.0, places=6)
        self.assertAlmostEqual(result.mean_slope_deg, 0.0, places=6)
        self.assertAlmostEqual(result.mean_z, 10.0, places=6)

    def test_forty_five_degree_roof_uses_three_dimensional_area(self):
        roof = Polygon([(0, 0, 0), (10, 0, 10), (10, 10, 10), (0, 10, 0)])

        result = measure_polygon_3d(roof)

        self.assertAlmostEqual(result.projected_area, 100.0, places=6)
        self.assertAlmostEqual(result.area_3d, 100.0 * math.sqrt(2), places=6)
        self.assertAlmostEqual(result.mean_slope_deg, 45.0, places=6)
        self.assertAlmostEqual(result.mean_z, 5.0, places=6)

    def test_g1_height_and_relief_use_ground_elevation(self):
        roof = Polygon([(0, 0, 8), (10, 0, 13), (10, 10, 13), (0, 10, 8)])
        ground = Polygon([(0, 0, 1), (10, 0, 1), (10, 10, 1), (0, 10, 1)])

        self.assertAlmostEqual(g1_height([roof], [ground]), 9.5, places=6)
        self.assertAlmostEqual(
            roof_relief_ratio([roof], [ground]), 5.0 / 12.0, places=6
        )


if __name__ == "__main__":
    unittest.main()
