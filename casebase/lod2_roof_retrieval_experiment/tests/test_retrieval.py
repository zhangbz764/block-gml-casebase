import unittest

import numpy as np
import pandas as pd

from retrieval import (
    RobustFeatureScaler,
    retrieve_for_query,
    select_development_queries,
)


class RobustScalerTests(unittest.TestCase):
    def test_uses_median_and_iqr_with_safe_constant_scale(self):
        frame = pd.DataFrame({"varying": [0, 1, 2, 3, 4], "constant": [7] * 5})

        scaler = RobustFeatureScaler.fit(frame, ["varying", "constant"])
        transformed = scaler.transform(frame)

        self.assertAlmostEqual(scaler.medians["varying"], 2.0)
        self.assertAlmostEqual(scaler.iqrs["varying"], 2.0)
        self.assertAlmostEqual(scaler.iqrs["constant"], 1.0)
        np.testing.assert_allclose(transformed["varying"], [-1, -0.5, 0, 0.5, 1])


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame(
            {
                "block_id": ["q", "a", "b", "c", "d", "e"],
                "general": [0.0, 1.0, 1.0, 2.0, 3.0, 4.0],
                "roof": [0.0, 10.0, 0.1, 0.0, 0.0, 0.0],
            }
        )

    def test_self_exclusion_shared_pool_and_roof_reranking(self):
        result = retrieve_for_query(
            self.frame,
            "q",
            general_features=["general"],
            roof_features=["roof"],
            pool_size=3,
            top_k=1,
            lambda_values=[0.8],
        )

        m1 = result[result["method"] == "M1"].iloc[0]
        m2 = result[result["method"] == "M2"].iloc[0]
        self.assertEqual(m1["candidate_id"], "a")
        self.assertEqual(m2["candidate_id"], "c")
        self.assertNotEqual(m1["candidate_id"], "q")
        self.assertEqual(m2["general_pool_rank"], 3)
        self.assertEqual(set(result["pool_size"]), {3})

    def test_returns_requested_top_k_for_each_method(self):
        result = retrieve_for_query(
            self.frame,
            "q",
            general_features=["general"],
            roof_features=["roof"],
            pool_size=4,
            top_k=2,
            lambda_values=[0.2, 0.5],
        )

        counts = result.groupby(["method", "lambda_value"]).size().to_dict()
        self.assertEqual(counts[("M1", 0.0)], 2)
        self.assertEqual(counts[("M2", 0.2)], 2)
        self.assertEqual(counts[("M2", 0.5)], 2)
        self.assertTrue(set(result["candidate_id"]).issubset({"a", "b", "c", "d"}))

    def test_preserves_query_and_candidate_city_metadata(self):
        frame = self.frame.assign(
            city=["luxembourg", "berlin", "luxembourg", "vienna", "berlin", "vienna"]
        )

        result = retrieve_for_query(
            frame,
            "q",
            general_features=["general"],
            roof_features=["roof"],
            pool_size=3,
            top_k=1,
            lambda_values=[0.8],
        )

        self.assertEqual(set(result["query_city"]), {"luxembourg"})
        self.assertEqual(result.iloc[0]["candidate_city"], "berlin")
        self.assertFalse(result.iloc[0]["same_city"])

    def test_city_disambiguates_duplicate_block_ids(self):
        frame = pd.DataFrame(
            {
                "block_id": ["q", "q", "a", "b"],
                "city": ["berlin", "luxembourg", "berlin", "vienna"],
                "general": [100.0, 0.0, 1.0, 2.0],
                "roof": [100.0, 0.0, 1.0, 2.0],
            }
        )

        result = retrieve_for_query(
            frame,
            "q",
            query_city="luxembourg",
            general_features=["general"],
            roof_features=["roof"],
            pool_size=3,
            top_k=1,
            lambda_values=[0.35],
        )

        self.assertEqual(set(result["query_city"]), {"luxembourg"})
        self.assertEqual(result.iloc[0]["candidate_id"], "a")
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            retrieve_for_query(
                frame,
                "q",
                general_features=["general"],
                roof_features=["roof"],
                pool_size=3,
                top_k=1,
                lambda_values=[0.35],
            )

    def test_development_queries_are_unique_and_named(self):
        frame = pd.DataFrame(
            {
                "block_id": [f"b{i}" for i in range(8)],
                "roof_mean_slope": [0, 5, 10, 15, 20, 25, 30, 35],
                "building_roof_slope_sd": [8, 7, 6, 5, 4, 3, 2, 1],
                "roof_relief_ratio_median": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            }
        )

        result = select_development_queries(frame)

        self.assertEqual(len(result), 5)
        self.assertEqual(result["block_id"].nunique(), 5)
        self.assertEqual(
            set(result["query_type"]),
            {
                "lowest_mean_slope",
                "highest_mean_slope",
                "median_slope_low_dispersion",
                "median_slope_high_dispersion",
                "highest_relief_ratio",
            },
        )


if __name__ == "__main__":
    unittest.main()
