from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


PHASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PHASE_DIR))

from analyze_phase4b import summarize_status


class AnalysisTests(unittest.TestCase):
    def test_part_summary_closes_statuses_and_reports_query_share(self):
        status = pd.DataFrame([
            {"sample_id": "a", "attempt": 1, "status": "completed", "query_seconds": 2, "encoding_seconds": 8, "runtime_seconds": 10, "output_bytes": 100, "excluded_building_count": 1},
            {"sample_id": "b", "attempt": 1, "status": "failed", "query_seconds": 1, "encoding_seconds": 4, "runtime_seconds": 5, "output_bytes": 0, "excluded_building_count": 0},
        ])
        result = summarize_status(status, planned=2, skipped=0)

        self.assertEqual(result["completed"], 1)
        self.assertEqual(result["failed"], 1)
        self.assertEqual(result["interrupted"], 0)
        self.assertEqual(result["query_share"], 0.2)


if __name__ == "__main__":
    unittest.main()
