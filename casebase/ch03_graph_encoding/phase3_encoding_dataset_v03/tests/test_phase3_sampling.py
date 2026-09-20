from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path


PHASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PHASE_DIR))

from sample_phase3 import Phase3SamplingConfig, _select_from_stratum_pools, select_phase3_samples


def candidate(city: str, number: int, building_count: int) -> dict:
    return {
        "city": city,
        "country": "Testland",
        "block_id": f"{city[:2].upper()}_{number:04d}",
        "lod1_building_count": building_count,
        "area_m2": float(1000 + number),
        "bcr": 0.25,
        "avg_nn_distance": 8.0,
    }


class Phase3SamplingTests(unittest.TestCase):
    def setUp(self):
        self.config = Phase3SamplingConfig(seed=20260920)

    def test_same_seed_is_deterministic_without_mutating_input(self):
        rows = [candidate("Osaka", index, index + 2) for index in range(80)]
        original = copy.deepcopy(rows)

        first = select_phase3_samples(rows, config=self.config)
        second = select_phase3_samples(list(reversed(rows)), config=self.config)

        self.assertEqual(
            [row["block_id"] for row in first["selected"]],
            [row["block_id"] for row in second["selected"]],
        )
        self.assertEqual(rows, original)

    def test_core_selects_ten_from_each_equal_frequency_stratum(self):
        rows = [candidate("Winnipeg", index, index + 2) for index in range(80)]

        result = select_phase3_samples(rows, config=self.config)
        core = [row for row in result["selected"] if "core" in row["sample_groups"]]

        self.assertEqual(len(core), 40)
        self.assertEqual(len({row["block_id"] for row in core}), 40)
        self.assertEqual(
            {stratum: sum(row["selection_stratum"] == stratum for row in core) for stratum in range(1, 5)},
            {1: 10, 2: 10, 3: 10, 4: 10},
        )

    def test_underfull_stratum_is_supplemented_from_nearest_stratum(self):
        pools = {
            1: [candidate("Portland", index, 2) for index in range(5)],
            2: [candidate("Portland", index, 3) for index in range(5, 30)],
            3: [candidate("Portland", index, 4) for index in range(30, 45)],
            4: [candidate("Portland", index, 5) for index in range(45, 60)],
        }

        selections, log = _select_from_stratum_pools(pools, "Portland", self.config)

        self.assertEqual(len(selections), 40)
        self.assertEqual(sum(item["selection_stratum"] == 1 for item in selections), 10)
        self.assertEqual(sum(item["supplemented"] for item in selections), 5)
        self.assertEqual(log, [{"city": "Portland", "target_stratum": 1, "deficit": 5, "supplied": 5, "unresolved": 0}])

    def test_stress_and_regression_memberships_survive_deduplication(self):
        rows = [candidate("Osaka", index, index + 2) for index in range(50)]
        rows.extend(
            [
                candidate("Osaka", 100, 1),
                candidate("Osaka", 101, 503),
                candidate("Osaka", 102, 520),
            ]
        )
        anchors = [
            {"sample_id": "anchor_core", "city": "Osaka", "block_id": "OS_0000"},
            {"sample_id": "anchor_large", "city": "Osaka", "block_id": "OS_0101"},
        ]

        result = select_phase3_samples(rows, regression_anchors=anchors, config=self.config)
        by_id = {row["block_id"]: row for row in result["selected"]}

        self.assertEqual(len(by_id), len(result["selected"]))
        self.assertIn("regression_anchor", by_id["OS_0000"]["sample_groups"])
        self.assertEqual(by_id["OS_0101"]["sample_groups"], ["stress_large", "regression_anchor"])
        self.assertEqual(by_id["OS_0101"]["regression_anchor_ids"], ["anchor_large"])
        self.assertIn("stress_single", by_id["OS_0100"]["sample_groups"])

    def test_frame_records_selected_and_not_selected_reasons(self):
        rows = [candidate("Winnipeg", index, index + 2) for index in range(60)]

        result = select_phase3_samples(rows, config=self.config)

        self.assertEqual(len(result["frame"]), 60)
        self.assertTrue(all(row["selection_reason"] for row in result["frame"]))
        self.assertEqual(sum(row["selected"] for row in result["frame"]), 40)
        self.assertTrue(any(row["selection_reason"] == "not_selected_core_quota" for row in result["frame"]))


if __name__ == "__main__":
    unittest.main()
