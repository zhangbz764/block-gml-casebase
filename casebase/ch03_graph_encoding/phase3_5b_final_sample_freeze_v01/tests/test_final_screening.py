from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


PHASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PHASE_DIR))

from freeze_final_sample import (
    apply_final_rules,
    build_final_summaries,
    build_graph_manifest,
    compute_p995_threshold,
    stable_csv_bytes,
)


def base_row(block_id: str, **updates) -> dict:
    row = {
        "city": "TestCity",
        "block_id": block_id,
        "source_relation": "summary.lod1_candidate_blocks",
        "rule_version": "0.1",
        "area_m2": 1000.0,
        "lod1_building_count": 100,
        "bcr": 0.3,
        "hard_flag_count": 0,
        "hard_reasons": "",
        "primary_hard_reason": "",
        **{f"H{i:02d}": False for i in range(1, 6)},
        **{f"S{i:02d}": False for i in range(1, 7)},
    }
    row.update(updates)
    return row


def decision_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        [
            base_row(
                "HARD",
                H03=True,
                hard_flag_count=1,
                hard_reasons="coverage_inconsistent",
                primary_hard_reason="coverage_inconsistent",
            ),
            base_row("S01", S01=True),
            base_row("S02_ONLY", S02=True),
            base_row("S03", S03=True),
            base_row("S04_368", S04=True, lod1_building_count=368),
            base_row("S04_369", S04=True, lod1_building_count=369),
            base_row("S05", S05=True),
            base_row("S06", S01=True, S05=True, S06=True),
            base_row("SCALE_NO_S04", lod1_building_count=400),
            base_row("NORMAL"),
        ]
    )


class FinalScreeningTests(unittest.TestCase):
    def test_frozen_rule_decisions_and_reason_priority(self):
        ledger = apply_final_rules(decision_fixture(), threshold=369.0).set_index("block_id")

        self.assertEqual(ledger.loc["HARD", "final_decision"], "exclude")
        self.assertEqual(ledger.loc["S01", "final_decision"], "exclude")
        self.assertEqual(ledger.loc["S02_ONLY", "final_decision"], "eligible")
        self.assertEqual(ledger.loc["S03", "final_decision"], "exclude")
        self.assertEqual(ledger.loc["S04_368", "final_decision"], "eligible")
        self.assertEqual(ledger.loc["S04_369", "final_decision"], "exclude")
        self.assertEqual(ledger.loc["S05", "final_decision"], "exclude")
        self.assertEqual(ledger.loc["S06", "final_decision"], "exclude")
        self.assertEqual(ledger.loc["SCALE_NO_S04", "final_decision"], "exclude")
        self.assertEqual(ledger.loc["NORMAL", "final_decision"], "eligible")
        self.assertEqual(ledger.loc["HARD", "primary_final_reason"], "coverage_inconsistent")
        self.assertEqual(ledger.loc["S06", "primary_final_reason"], "area_upper_tail_p99")
        self.assertEqual(
            ledger.loc["S06", "final_exclusion_reasons"].split(";").count("large_sparse_combination"),
            1,
        )

    def test_graph_scale_rule_applies_only_to_hard_eligible_rows(self):
        frame = pd.DataFrame(
            [
                base_row("ELIGIBLE_HIGH", lod1_building_count=369),
                base_row(
                    "HARD_HIGH",
                    lod1_building_count=999,
                    H03=True,
                    hard_flag_count=1,
                    hard_reasons="coverage_inconsistent",
                    primary_hard_reason="coverage_inconsistent",
                ),
            ]
        )

        ledger = apply_final_rules(frame, threshold=369.0).set_index("block_id")

        self.assertTrue(ledger.loc["ELIGIBLE_HIGH", "exclude_graph_scale_p995"])
        self.assertFalse(ledger.loc["HARD_HIGH", "exclude_graph_scale_p995"])

    def test_p995_threshold_uses_only_hard_eligible_rows(self):
        rows = [base_row(f"B{i:03d}", lod1_building_count=1) for i in range(198)]
        rows.extend(
            [
                base_row("TOP1", lod1_building_count=369),
                base_row("TOP2", lod1_building_count=369),
                base_row("HARD_OUTLIER", lod1_building_count=9999, H01=True, hard_flag_count=1),
            ]
        )

        self.assertEqual(compute_p995_threshold(pd.DataFrame(rows)), 369.0)

    def test_result_and_hash_are_deterministic_under_input_shuffle(self):
        frame = decision_fixture()
        shuffled = frame.sample(frac=1.0, random_state=17).reset_index(drop=True)

        first = apply_final_rules(frame, threshold=369.0)
        second = apply_final_rules(shuffled, threshold=369.0)

        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(stable_csv_bytes(first), stable_csv_bytes(second))

    def test_graph_manifest_contains_only_unique_eligible_blocks(self):
        ledger = apply_final_rules(decision_fixture(), threshold=369.0)

        manifest = build_graph_manifest(ledger)

        expected = set(
            ledger.loc[ledger["final_decision"].eq("eligible"), ["city", "block_id"]]
            .itertuples(index=False, name=None)
        )
        actual = set(manifest[["city", "block_id"]].itertuples(index=False, name=None))
        self.assertEqual(actual, expected)
        self.assertFalse(manifest[["city", "block_id"]].duplicated().any())
        self.assertTrue(manifest["graph_encoding_eligible"].all())
        self.assertEqual(
            manifest.columns.tolist(),
            [
                "city", "block_id", "lod1_building_count", "area_m2", "bcr",
                "final_rule_version", "graph_encoding_eligible", "source_run_id",
            ],
        )

    def test_summaries_close_and_s06_adds_no_second_exclusion(self):
        ledger = apply_final_rules(decision_fixture(), threshold=369.0)

        summaries = build_final_summaries(ledger)
        flow = summaries["final_filter_flow"]
        city = summaries["final_summary_by_city"]

        self.assertEqual(int(flow.iloc[-1]["cumulative_excluded_count"]), 7)
        self.assertEqual(int(flow.iloc[-1]["remaining_count"]), 3)
        s06 = flow.loc[flow["rule_id"].eq("S06")].iloc[0]
        self.assertEqual(int(s06["newly_excluded_count"]), 0)
        self.assertEqual(int(city["input_count"].sum()), 10)
        self.assertEqual(int(city["excluded_count"].sum() + city["eligible_count"].sum()), 10)


if __name__ == "__main__":
    unittest.main()
