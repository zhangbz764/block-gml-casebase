from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


PHASE_DIR = Path(__file__).resolve().parents[1]
TEST_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PHASE_DIR))
sys.path.insert(0, str(TEST_DIR))

from fixtures import valid_row
from screen_blocks import compute_hard_flags, compute_review_thresholds, screen_dataframe


class ScreeningRuleTests(unittest.TestCase):
    def test_single_building_expected_nulls_are_kept(self):
        row = valid_row(lod1_building_count=1)
        for field in ("hsd", "hcv", "footprint_cv", "avg_nn_distance"):
            row[field] = None

        ledger = screen_dataframe(pd.DataFrame([row]))

        self.assertEqual(ledger.loc[0, "machine_status"], "keep_provisional")
        self.assertFalse(bool(ledger.loc[0, "H04"]))

    def test_multibuilding_null_is_hard_exclusion(self):
        row = valid_row(lod1_building_count=2, avg_nn_distance=None)

        ledger = screen_dataframe(pd.DataFrame([row]))

        self.assertTrue(bool(ledger.loc[0, "H04"]))
        self.assertEqual(ledger.loc[0, "machine_status"], "exclude_hard")

    def test_duplicate_keys_are_flagged_as_invalid(self):
        frame = pd.DataFrame([valid_row(), valid_row()])

        flags = compute_hard_flags(frame)

        self.assertEqual(flags["H01"].tolist(), [True, True])

    def test_invalid_geometry_is_hard_exclusion(self):
        ledger = screen_dataframe(pd.DataFrame([valid_row(geom_is_valid=False)]))

        self.assertTrue(bool(ledger.loc[0, "H01"]))
        self.assertEqual(ledger.loc[0, "primary_hard_reason"], "key_or_block_geometry_invalid")

    def test_existing_qa_error_boundary_is_h02(self):
        ledger = screen_dataframe(pd.DataFrame([valid_row(bcd=0.0)]))

        self.assertTrue(bool(ledger.loc[0, "H02"]))
        self.assertIn("nonpositive_bcd", ledger.loc[0, "H02_qa_reasons"])

    def test_multibuilding_zero_nn_is_h05(self):
        ledger = screen_dataframe(pd.DataFrame([valid_row(avg_nn_distance=0.0)]))

        self.assertTrue(bool(ledger.loc[0, "H05"]))
        self.assertEqual(ledger.loc[0, "primary_hard_reason"], "duplicate_centroid_signal")

    def test_thresholds_ignore_hard_exclusions(self):
        rows = [valid_row(block_id=f"B{i:03d}", area_m2=100.0 + i) for i in range(100)]
        rows.append(valid_row(block_id="BAD", area_m2=1_000_000_000.0, bcd=0.0))
        frame = pd.DataFrame(rows)
        flags = compute_hard_flags(frame)
        eligible = frame.loc[flags["hard_flag_count"].eq(0)]

        thresholds = compute_review_thresholds(eligible)

        self.assertLess(thresholds["area_m2_p99"], 200.0)
        self.assertEqual(thresholds["eligible_count"], 100)


if __name__ == "__main__":
    unittest.main()
