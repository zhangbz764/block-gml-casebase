from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


PHASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PHASE_DIR))

from build_solo_flag_review import (
    EXPECTED_POOL_COUNTS,
    build_solo_manifest,
    build_solo_pools,
    output_paths,
    verification_summary,
)


def solo_fixture() -> pd.DataFrame:
    rows = []
    for rule, metric in (("S02", "elongation"), ("S04", "lod1_building_count")):
        for index in range(30):
            row = {
                "city": f"City{index % 8}",
                "block_id": f"{rule}_{index:03d}",
                "elongation": 0.01 + index / 1000,
                "lod1_building_count": 500 - index,
                "hard_flag_count": 0,
                "hard_reasons": "",
                "scope_reasons": rule,
                **{f"H{i:02d}": False for i in range(1, 6)},
                **{f"S{i:02d}": False for i in range(1, 7)},
            }
            row[rule] = True
            rows.append(row)

    contaminated = dict(rows[0])
    contaminated["block_id"] = "S02_WITH_S01"
    contaminated["S01"] = True
    rows.append(contaminated)

    hard = dict(rows[1])
    hard["block_id"] = "S02_WITH_HARD"
    hard["H03"] = True
    hard["hard_flag_count"] = 1
    rows.append(hard)
    return pd.DataFrame(rows)


def exhausted_city_head_fixture() -> pd.DataFrame:
    rows = []
    for rule in ("S02", "S04"):
        for city_index in range(4):
            for rank, suffix in enumerate(("A", "B", "C", "D")):
                row = {
                    "city": f"City{city_index}",
                    "block_id": f"{rule}_{city_index}_{suffix}",
                    "elongation": 0.5,
                    "lod1_building_count": 100,
                    "hard_flag_count": 0,
                    **{f"H{i:02d}": False for i in range(1, 6)},
                    **{f"S{i:02d}": False for i in range(1, 7)},
                }
                row[rule] = True
                if rule == "S02":
                    row["elongation"] = rank + city_index / 100
                else:
                    row["lod1_building_count"] = 1000 - rank * 100 - city_index
                rows.append(row)
    return pd.DataFrame(rows)


class SoloFlagReviewTests(unittest.TestCase):
    def test_solo_pools_exclude_other_soft_and_all_hard_flags(self):
        pools = build_solo_pools(solo_fixture())

        self.assertEqual(len(pools["S02_only"]), 30)
        self.assertEqual(len(pools["S04_only"]), 30)
        self.assertNotIn("S02_WITH_S01", set(pools["S02_only"]["block_id"]))
        self.assertNotIn("S02_WITH_HARD", set(pools["S02_only"]["block_id"]))
        for solo_rule, pool in pools.items():
            target = solo_rule[:3]
            other_rules = [f"S{i:02d}" for i in range(1, 7) if f"S{i:02d}" != target]
            self.assertTrue(pool[target].all())
            self.assertFalse(pool[other_rules].any(axis=None))
            self.assertTrue(pool["hard_flag_count"].eq(0).all())

    def test_manifest_has_twelve_unique_cases_per_rule_and_all_roles(self):
        manifest = build_solo_manifest(solo_fixture(), per_rule=12, source_run_id="full_v01")

        self.assertEqual(len(manifest), 24)
        self.assertFalse(manifest[["city", "block_id"]].duplicated().any())
        self.assertEqual(manifest.groupby("solo_rule").size().to_dict(), {"S02_only": 12, "S04_only": 12})
        self.assertEqual(
            set(manifest["review_sample_roles"]),
            {"extreme", "near_threshold", "city_rotation"},
        )
        self.assertTrue(manifest["review_case_type"].eq("solo_flag_review").all())
        self.assertTrue(manifest["source_run_id"].eq("full_v01").all())

    def test_manifest_is_deterministic_under_input_shuffle(self):
        frame = solo_fixture()
        shuffled = frame.sample(frac=1.0, random_state=42).reset_index(drop=True)

        first = build_solo_manifest(frame, per_rule=12, source_run_id="full_v01")
        second = build_solo_manifest(shuffled, per_rule=12, source_run_id="full_v01")

        pd.testing.assert_frame_equal(first, second)

    def test_city_rotation_uses_each_city_after_prior_roles_are_removed(self):
        manifest = build_solo_manifest(
            exhausted_city_head_fixture(),
            per_rule=12,
            source_run_id="full_v01",
        )

        rotation = manifest.loc[manifest["review_sample_roles"].eq("city_rotation")]
        for _, group in rotation.groupby("solo_rule"):
            self.assertEqual(group["city"].nunique(), 4)

    def test_full_ledger_pool_counts_match_frozen_expectations(self):
        ledger_path = PHASE_DIR / "results" / "bulk" / "full_v01" / "block_screening_ledger.csv.gz"
        if not ledger_path.exists():
            self.skipTest("ignored full_v01 ledger is not available in this checkout")
        ledger = pd.read_csv(ledger_path, encoding="utf-8-sig", low_memory=False)

        counts = {name: len(pool) for name, pool in build_solo_pools(ledger).items()}

        self.assertEqual(counts, EXPECTED_POOL_COUNTS)

    def test_output_paths_are_separate_from_source_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = output_paths(root, source_run_id="full_v01", output_run_id="full_v01_solo")

            self.assertTrue(all("full_v01_solo" in str(path) for path in paths.values()))
            self.assertTrue(all(path.parent.name == "full_v01_solo" for path in paths.values()))
            with self.assertRaises(ValueError):
                output_paths(root, source_run_id="full_v01", output_run_id="full_v01")

    def test_verification_summary_records_counts_roles_and_passed(self):
        manifest = build_solo_manifest(solo_fixture(), per_rule=12, source_run_id="full_v01")
        summary = verification_summary(
            {"S02_only": 1049, "S04_only": 1184},
            manifest,
            per_rule=12,
        )

        self.assertTrue(summary["passed"])
        self.assertEqual(summary["candidate_pool_counts"], EXPECTED_POOL_COUNTS)
        self.assertEqual(summary["sample_counts"], {"S02_only": 12, "S04_only": 12})
        self.assertEqual(summary["unique_block_count"], 24)
        self.assertEqual(
            summary["role_counts"],
            {"city_rotation": 8, "extreme": 8, "near_threshold": 8},
        )


if __name__ == "__main__":
    unittest.main()
