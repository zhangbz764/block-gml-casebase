import unittest

import numpy as np
import pandas as pd

try:
    from evaluation import (
        evaluate_roof_distributions,
        select_formal_queries,
        summarize_formal_evaluation,
    )
except ModuleNotFoundError:
    evaluate_roof_distributions = None
    select_formal_queries = None
    summarize_formal_evaluation = None


class FormalQueryTests(unittest.TestCase):
    def setUp(self):
        count = 52
        self.blocks = pd.DataFrame(
            {
                "block_id": [f"b{i:02d}" for i in range(count)],
                "roof_mean_slope": np.linspace(0, 51, count),
                "roof_relief_ratio_median": np.linspace(0.05, 0.56, count),
                "building_roof_slope_sd": np.linspace(1, 27, count),
                "is_usable": True,
            }
        )
        self.development_ids = {"b00", "b13", "b26", "b39", "b51"}

    def test_selects_disjoint_stratified_and_boundary_queries(self):
        self.assertIsNotNone(select_formal_queries)
        result = select_formal_queries(
            self.blocks, self.development_ids, seed=20260908, regular_count=16
        )

        self.assertEqual(len(result), 20)
        self.assertEqual(result["block_id"].nunique(), 20)
        self.assertTrue(set(result["block_id"]).isdisjoint(self.development_ids))
        regular = result[result["query_role"] == "regular"]
        boundary = result[result["query_role"] == "boundary"]
        self.assertEqual(
            regular.groupby("stratum").size().to_dict(),
            {"slope_q1": 4, "slope_q2": 4, "slope_q3": 4, "slope_q4": 4},
        )
        self.assertEqual(
            set(boundary["stratum"]),
            {"near_flat", "steep", "high_relief", "high_dispersion"},
        )

    def test_selection_is_deterministic(self):
        self.assertIsNotNone(select_formal_queries)
        first = select_formal_queries(self.blocks, self.development_ids)
        second = select_formal_queries(self.blocks, self.development_ids)
        pd.testing.assert_frame_equal(first, second)


class RoofDistributionTests(unittest.TestCase):
    def setUp(self):
        self.buildings = pd.DataFrame(
            {
                "block_id": ["q", "q", "a", "a", "b", "b"],
                "roof_mean_slope": [0.0, 10.0, 2.0, 12.0, 20.0, 30.0],
                "roof_relief_ratio": [0.1, 0.3, 0.2, 0.4, 0.6, 0.8],
                "is_usable": True,
            }
        )
        self.retrieval = pd.DataFrame(
            {
                "query_id": ["q", "q"],
                "method": ["M1", "M2"],
                "lambda_value": [0.0, 0.35],
                "rank": [1, 1],
                "candidate_id": ["b", "a"],
                "d_general": [1.0, 1.2],
                "d_roof": [2.0, 0.5],
                "d_combined": [1.0, 0.955],
                "pool_size": [2, 2],
                "general_pool_rank": [1, 2],
            }
        )

    def test_calculates_literal_distribution_distances(self):
        self.assertIsNotNone(evaluate_roof_distributions)
        result = evaluate_roof_distributions(self.buildings, self.retrieval)
        m1 = result[result["method"] == "M1"].iloc[0]
        m2 = result[result["method"] == "M2"].iloc[0]

        self.assertAlmostEqual(m1["slope_wasserstein_deg"], 20.0)
        self.assertAlmostEqual(m2["slope_wasserstein_deg"], 2.0)
        self.assertAlmostEqual(m1["relief_wasserstein"], 0.5)
        self.assertAlmostEqual(m2["relief_wasserstein"], 0.1)
        self.assertAlmostEqual(m1["flat_roof_share_abs_diff"], 0.5)
        self.assertAlmostEqual(m2["flat_roof_share_abs_diff"], 0.0)

    def test_summarizes_paired_m2_minus_m1_without_composite_score(self):
        self.assertIsNotNone(summarize_formal_evaluation)
        enriched = evaluate_roof_distributions(self.buildings, self.retrieval)
        query_summary, aggregate = summarize_formal_evaluation(enriched)

        self.assertEqual(len(query_summary), 2)
        m2_query = query_summary[query_summary["method"] == "M2"].iloc[0]
        self.assertAlmostEqual(m2_query["d_general_delta_vs_m1"], 0.2)
        self.assertAlmostEqual(m2_query["slope_wasserstein_deg_delta_vs_m1"], -18.0)
        m2_aggregate = aggregate[aggregate["method"] == "M2"].iloc[0]
        self.assertAlmostEqual(
            m2_aggregate["slope_wasserstein_deg_delta_vs_m1_mean"], -18.0
        )
        self.assertNotIn("composite_score", query_summary.columns)
        self.assertNotIn("composite_score", aggregate.columns)

    def test_missing_block_building_data_raises(self):
        self.assertIsNotNone(evaluate_roof_distributions)
        missing = self.retrieval.copy()
        missing.loc[0, "candidate_id"] = "missing"
        with self.assertRaisesRegex(ValueError, "missing"):
            evaluate_roof_distributions(self.buildings, missing)


if __name__ == "__main__":
    unittest.main()
