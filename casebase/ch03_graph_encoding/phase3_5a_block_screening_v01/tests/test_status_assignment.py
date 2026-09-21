from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


PHASE_DIR = Path(__file__).resolve().parents[1]
TEST_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PHASE_DIR))
sys.path.insert(0, str(TEST_DIR))

from fixtures import mixed_fixture, valid_row
from screen_blocks import screen_dataframe


class StatusAssignmentTests(unittest.TestCase):
    def test_bcr_and_osr_overlap_has_one_primary_reason(self):
        row = valid_row(bcr=1.2, osr=-0.1, far=2.0)

        ledger = screen_dataframe(pd.DataFrame([row]))

        self.assertTrue(bool(ledger.loc[0, "H03"]))
        self.assertEqual(ledger.loc[0, "primary_hard_reason"], "coverage_inconsistent")
        self.assertEqual(int(ledger.loc[0, "hard_flag_count"]), 1)

    def test_scope_tail_never_auto_excludes(self):
        ledger = screen_dataframe(mixed_fixture())
        extreme = ledger.loc[ledger["block_id"].eq("EXTREME")].iloc[0]

        self.assertGreaterEqual(int(extreme["scope_flag_count"]), 1)
        self.assertEqual(extreme["machine_status"], "review_scope")

    def test_statuses_are_mutually_exclusive_and_exhaustive(self):
        fixture = mixed_fixture()

        ledger = screen_dataframe(fixture)

        self.assertLessEqual(
            set(ledger["machine_status"]),
            {"exclude_hard", "review_scope", "keep_provisional"},
        )
        self.assertTrue(ledger["machine_status"].notna().all())
        self.assertEqual(len(ledger), len(fixture))

    def test_rule_overlap_preserves_all_reasons_and_priority(self):
        row = valid_row(bcd=0.0, bcr=1.2, osr=-0.1, far=2.0, avg_nn_distance=0.0)

        ledger = screen_dataframe(pd.DataFrame([row]))

        self.assertEqual(int(ledger.loc[0, "hard_flag_count"]), 3)
        self.assertEqual(ledger.loc[0, "primary_hard_reason"], "candidate_contract_broken")
        self.assertEqual(
            ledger.loc[0, "hard_reasons"],
            "candidate_contract_broken;coverage_inconsistent;duplicate_centroid_signal",
        )


if __name__ == "__main__":
    unittest.main()
