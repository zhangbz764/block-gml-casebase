from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


PHASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PHASE_DIR))

from build_s04_extended_review import (
    EXPECTED_S04_POOL_SIZE,
    build_extended_manifest,
    build_manual_review,
    build_sampling_summary,
    output_paths,
    verification_summary,
)
from build_solo_flag_review import build_solo_pools


def extended_fixture() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    cell_first_keys = []
    row_index = 0
    for count_quartile in range(4):
        for area_quartile in range(4):
            for repeat in range(4):
                city = f"City{row_index % 20:02d}"
                block_id = f"B{row_index:04d}"
                row = {
                    "city": city,
                    "block_id": block_id,
                    "area_m2": 1000.0 * (area_quartile + 1) + repeat,
                    "lod1_building_count": 100 * (count_quartile + 1) + repeat,
                    "hard_flag_count": 0,
                    "hard_reasons": "",
                    "scope_reasons": "building_count_upper_tail",
                    **{f"H{i:02d}": False for i in range(1, 6)},
                    **{f"S{i:02d}": False for i in range(1, 7)},
                }
                row["S04"] = True
                rows.append(row)
                if repeat == 0:
                    cell_first_keys.append((city, block_id))
                row_index += 1
    ledger = pd.DataFrame(rows)
    prior_keys = cell_first_keys[:12]
    prior = pd.DataFrame(
        [
            {
                "city": city,
                "block_id": block_id,
                "solo_rule": "S04_only",
                "review_sample_roles": "prior_role",
            }
            for city, block_id in prior_keys
        ]
    )
    return ledger, prior


class S04ExtendedReviewTests(unittest.TestCase):
    def test_manifest_keeps_prior_twelve_and_adds_forty_eight_unique_cases(self):
        ledger, prior = extended_fixture()

        manifest = build_extended_manifest(ledger, prior, new_sample_size=48, seed=20260922)

        self.assertEqual(len(manifest), 60)
        self.assertFalse(manifest[["city", "block_id"]].duplicated().any())
        self.assertEqual(manifest["sample_origin"].eq("prior_12").sum(), 12)
        self.assertEqual(manifest["sample_origin"].ne("prior_12").sum(), 48)
        expected_prior = list(prior[["city", "block_id"]].itertuples(index=False, name=None))
        actual_prior = list(
            manifest.loc[manifest["sample_origin"].eq("prior_12"), ["city", "block_id"]]
            .itertuples(index=False, name=None)
        )
        self.assertEqual(actual_prior, expected_prior)
        self.assertEqual(manifest["extended_case_no"].tolist(), list(range(1, 61)))

    def test_metric_grid_has_two_new_cases_in_each_of_sixteen_cells(self):
        ledger, prior = extended_fixture()

        manifest = build_extended_manifest(ledger, prior, new_sample_size=48, seed=20260922)
        grid = manifest.loc[manifest["sample_origin"].eq("metric_grid")]

        self.assertEqual(len(grid), 32)
        self.assertEqual(len(grid.groupby(["count_band", "area_band"])), 16)
        self.assertTrue(grid.groupby(["count_band", "area_band"]).size().eq(2).all())
        self.assertEqual(set(manifest["count_band"]), {"Q1", "Q2", "Q3", "Q4"})
        self.assertEqual(set(manifest["area_band"]), {"Q1", "Q2", "Q3", "Q4"})

    def test_city_completion_covers_all_twenty_candidate_cities(self):
        ledger, prior = extended_fixture()

        manifest = build_extended_manifest(ledger, prior, new_sample_size=48, seed=20260922)

        self.assertEqual(manifest["city"].nunique(), 20)
        self.assertEqual(set(manifest["city"]), set(ledger["city"]))

    def test_same_seed_is_deterministic_under_ledger_shuffle(self):
        ledger, prior = extended_fixture()
        shuffled = ledger.sample(frac=1.0, random_state=91).reset_index(drop=True)

        first = build_extended_manifest(ledger, prior, new_sample_size=48, seed=20260922)
        second = build_extended_manifest(shuffled, prior, new_sample_size=48, seed=20260922)

        pd.testing.assert_frame_equal(first, second)

    def test_manual_review_is_blank_and_sampling_summary_has_required_dimensions(self):
        ledger, prior = extended_fixture()
        manifest = build_extended_manifest(ledger, prior, new_sample_size=48, seed=20260922)

        manual = build_manual_review(manifest)
        summary = build_sampling_summary(manifest)

        self.assertEqual(len(manual), 60)
        self.assertTrue(manual[["manual_decision", "manual_reason", "manual_notes"]].eq("").all(axis=None))
        self.assertEqual(
            manual.columns.tolist(),
            [
                "extended_case_no", "city", "block_id", "sample_origin",
                "count_band", "area_band", "manual_decision", "manual_reason", "manual_notes",
            ],
        )
        self.assertTrue({"city", "count_band", "area_band", "metric_grid"}.issubset(set(summary["dimension"])))

    def test_verification_summary_records_acceptance_counts(self):
        ledger, prior = extended_fixture()
        manifest = build_extended_manifest(ledger, prior, new_sample_size=48, seed=20260922)
        prior_keys = set(prior[["city", "block_id"]].itertuples(index=False, name=None))

        summary = verification_summary(
            pool_size=EXPECTED_S04_POOL_SIZE,
            candidate_city_count=20,
            manifest=manifest,
            prior_keys=prior_keys,
            new_sample_size=48,
        )

        self.assertTrue(summary["passed"])
        self.assertEqual(summary["candidate_pool_count"], 1184)
        self.assertEqual(summary["sample_count"], 60)
        self.assertEqual(summary["prior_sample_count"], 12)
        self.assertEqual(summary["new_sample_count"], 48)
        self.assertEqual(summary["candidate_city_count"], 20)
        self.assertEqual(summary["sampled_city_count"], 20)
        self.assertEqual(summary["metric_grid_cell_count"], 16)
        self.assertGreaterEqual(summary["metric_grid_min_per_cell"], 2)

    def test_full_ledger_still_has_frozen_s04_only_pool_size(self):
        ledger_path = PHASE_DIR / "results" / "bulk" / "full_v01" / "block_screening_ledger.csv.gz"
        if not ledger_path.exists():
            self.skipTest("ignored full_v01 ledger is not available in this checkout")
        ledger = pd.read_csv(ledger_path, encoding="utf-8-sig", low_memory=False)

        self.assertEqual(len(build_solo_pools(ledger)["S04_only"]), EXPECTED_S04_POOL_SIZE)

    def test_output_paths_cannot_reuse_source_or_prior_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = output_paths(
                root,
                source_run_id="full_v01",
                prior_run_id="full_v01_solo",
                output_run_id="full_v01_s04_extended",
            )

            self.assertTrue(all(path.parent.name == "full_v01_s04_extended" for path in paths.values()))
            for invalid in ("full_v01", "full_v01_solo"):
                with self.assertRaises(ValueError):
                    output_paths(
                        root,
                        source_run_id="full_v01",
                        prior_run_id="full_v01_solo",
                        output_run_id=invalid,
                    )


if __name__ == "__main__":
    unittest.main()
