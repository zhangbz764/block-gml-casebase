from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


PHASE_DIR = Path(__file__).resolve().parents[1]
TEST_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PHASE_DIR))
sys.path.insert(0, str(TEST_DIR))

from fixtures import mixed_fixture, valid_row
from run_phase3_5a import _write_csv, run_output_paths, select_pilot_frame, validate_run_id
from screen_blocks import build_filter_summaries, screen_dataframe
from verify_screening import (
    artifact_git_policy,
    pilot_recompute_matches,
    single_expected_nulls_are_preserved,
)


class SummaryTests(unittest.TestCase):
    def test_full_run_collision_scope_includes_all_run_directories(self):
        paths = {path.as_posix() for path in run_output_paths("full", "full_v01")}

        self.assertTrue(any(path.endswith("bulk/full_v01") for path in paths))
        self.assertTrue(any(path.endswith("audit/full_v01") for path in paths))
        self.assertTrue(any(path.endswith("visual_review/full_v01") for path in paths))

    def test_run_id_rejects_path_traversal(self):
        with self.assertRaises(ValueError):
            validate_run_id("../full_v01")

    def test_output_inventory_marks_regenerable_large_artifacts_ignored(self):
        self.assertEqual(artifact_git_policy(Path("bulk/full_v01/ledger.csv.gz")), "ignored")
        self.assertEqual(artifact_git_policy(Path("visual_review/full_v01/review_cases.html")), "ignored")
        self.assertEqual(artifact_git_policy(Path("audit/full_v01/filter_flow_summary.csv")), "tracked")

    def test_single_building_verification_allows_independent_unexpected_missing(self):
        expected = valid_row(block_id="EXPECTED", lod1_building_count=1)
        for field in ("hsd", "hcv", "footprint_cv", "avg_nn_distance"):
            expected[field] = None
        unexpected = valid_row(block_id="UNEXPECTED", lod1_building_count=1, avg_complexity=None)
        ledger = screen_dataframe(pd.DataFrame([expected, unexpected]))

        self.assertTrue(single_expected_nulls_are_preserved(ledger))
        self.assertTrue(bool(ledger.loc[ledger["block_id"].eq("UNEXPECTED"), "H04"].iloc[0]))

    def test_pilot_hash_recompute_uses_round_trip_float_parsing(self):
        row = valid_row(bcr=0.000788239007759977, far=1.0)
        row["osr"] = (1.0 - row["bcr"]) / row["far"]
        manifest = pd.DataFrame([{**row, "selection_reasons": "city_head10"}])
        ledger = screen_dataframe(manifest)
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = Path(directory) / "manifest.csv"
            ledger_path = Path(directory) / "ledger.csv"
            _write_csv(manifest_path, manifest)
            _write_csv(ledger_path, ledger)

            self.assertTrue(pilot_recompute_matches(manifest_path, ledger_path))

    def test_pilot_selection_unions_city_baseline_and_known_anomalies(self):
        rows = []
        for city in ("A", "B"):
            rows.extend(
                valid_row(city=city, block_id=f"{city}{index:02d}")
                for index in range(12)
            )
        rows[11].update({"bcr": 1.2, "far": 2.0, "osr": -0.1})
        rows[23].update({"avg_nn_distance": 0.0})
        rows[10].update({"avg_nn_distance": None})

        pilot = select_pilot_frame(pd.DataFrame(rows), per_city=10)

        self.assertEqual(len(pilot), 23)
        self.assertEqual(pilot.groupby("city").size().to_dict(), {"A": 12, "B": 11})
        self.assertIn("coverage_inconsistent", pilot.loc[pilot["block_id"].eq("A11"), "selection_reasons"].iloc[0])
        self.assertIn("unexpected_metric_missing", pilot.loc[pilot["block_id"].eq("A10"), "selection_reasons"].iloc[0])
        self.assertIn("duplicate_centroid_signal", pilot.loc[pilot["block_id"].eq("B11"), "selection_reasons"].iloc[0])

    def test_input_order_does_not_change_ledger_or_summaries(self):
        frame = mixed_fixture()
        shuffled = frame.sample(frac=1.0, random_state=7).reset_index(drop=True)

        first = screen_dataframe(frame)
        second = screen_dataframe(shuffled)
        first_summaries = build_filter_summaries(first)
        second_summaries = build_filter_summaries(second)

        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(set(first_summaries), set(second_summaries))
        for name in first_summaries:
            pd.testing.assert_frame_equal(first_summaries[name], second_summaries[name])

    def test_flow_and_city_summaries_close(self):
        frame = pd.DataFrame(
            [
                valid_row(city="A", block_id="A1"),
                valid_row(city="A", block_id="A2", bcd=0.0),
                valid_row(city="B", block_id="B1", avg_nn_distance=0.0),
            ]
        )

        ledger = screen_dataframe(frame)
        summaries = build_filter_summaries(ledger)
        flow = summaries["filter_flow_summary"]
        city = summaries["filter_summary_by_city"]

        self.assertEqual(int(flow.iloc[-1]["cumulative_excluded_count"]), 2)
        self.assertEqual(int(flow.iloc[-1]["remaining_count"]), 1)
        self.assertEqual(int(city["input_count"].sum()), len(ledger))
        self.assertEqual(
            int(city[["exclude_hard_count", "review_scope_count", "keep_provisional_count"]].sum().sum()),
            len(ledger),
        )

    def test_city_summary_keeps_missing_city_hard_error_in_closure(self):
        frame = pd.DataFrame(
            [valid_row(city=None, block_id="MISSING_CITY"), valid_row(city="A", block_id="A1")]
        )

        city = build_filter_summaries(screen_dataframe(frame))["filter_summary_by_city"]

        self.assertIn("(missing)", city["city"].tolist())
        self.assertEqual(
            int(city[["exclude_hard_count", "review_scope_count", "keep_provisional_count"]].sum().sum()),
            2,
        )


if __name__ == "__main__":
    unittest.main()
