from __future__ import annotations

import sys
import unittest
from pathlib import Path


PHASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PHASE_DIR))

from run_phase3 import select_pilot_rows


class Phase3PilotSelectionTests(unittest.TestCase):
    def test_pilot_uses_one_core_per_city_stratum_plus_deduplicated_anchors(self):
        rows = []
        for city in ("A", "B"):
            for stratum in range(1, 5):
                for duplicate in range(2):
                    rows.append(
                        {
                            "sample_id": f"{city}-{stratum}-{duplicate}",
                            "city": city,
                            "block_id": f"{city}-{stratum}-{duplicate}",
                            "selection_stratum": str(stratum),
                            "sample_groups": "core_audit",
                        }
                    )
        rows[0]["sample_groups"] = "core_audit;regression_anchor"
        rows.append(
            {
                "sample_id": "anchor-only",
                "city": "A",
                "block_id": "anchor-only",
                "selection_stratum": "",
                "sample_groups": "regression_anchor",
            }
        )

        selected = select_pilot_rows(rows)

        self.assertEqual(len(selected), 9)
        self.assertEqual(len({row["block_id"] for row in selected}), 9)
        self.assertEqual(sum("regression_anchor" in row["sample_groups"] for row in selected), 2)


if __name__ == "__main__":
    unittest.main()
