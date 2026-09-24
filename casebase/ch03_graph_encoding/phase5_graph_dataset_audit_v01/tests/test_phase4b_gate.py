from __future__ import annotations

import sys
import unittest
from pathlib import Path


PHASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PHASE_DIR))

from phase4b_gate import validate_phase4b_gate
from registry import load_audit_spec


SPEC = load_audit_spec(PHASE_DIR / "audit_spec_v01.json")


def passing_summary() -> dict:
    return {
        "passed": True,
        "missing_parts": [],
        "problems": {},
        "accounted_entries": 213054,
        "completed_graphs": 213000,
        "excluded_no_valid_building": 54,
        "expected_entries": 213054,
    }


class Phase4BGateTests(unittest.TestCase):
    def test_missing_part_blocks_phase5(self):
        summary = {**passing_summary(), "passed": False, "missing_parts": ["part_12"]}
        issues = validate_phase4b_gate(summary, SPEC)
        self.assertTrue(any("part_12" in issue for issue in issues))

    def test_unclosed_or_wrong_count_blocks_phase5(self):
        summary = {**passing_summary(), "passed": False, "problems": {"part_04": ["failed cases"]}, "accounted_entries": 213053}
        issues = validate_phase4b_gate(summary, SPEC)
        self.assertGreaterEqual(len(issues), 3)

    def test_only_closed_full_population_passes(self):
        self.assertEqual(validate_phase4b_gate(passing_summary(), SPEC), [])


if __name__ == "__main__":
    unittest.main()
