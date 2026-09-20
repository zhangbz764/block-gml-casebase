from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path


PHASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PHASE_DIR))

from audit_phase3 import build_feature_registry, numeric_profile, spearman_pairwise


class Phase3AuditTests(unittest.TestCase):
    def test_core_registry_excludes_source_leakage_and_quality_fields(self):
        registry = build_feature_registry()
        core = [row for row in registry if row["role"] == "core_model"]

        forbidden = ("city", "country", "srid", "source", "wkt", "quality", "failure", "candidate")
        self.assertFalse(
            [row["field"] for row in core if any(token in row["field"].lower() for token in forbidden)]
        )
        self.assertTrue(all(row["default_input"] == "yes" for row in core))

    def test_zero_is_not_missing_in_numeric_profile(self):
        profile = numeric_profile([0.0, None, 2.0, float("nan")])

        self.assertEqual(profile["valid_count"], 2)
        self.assertEqual(profile["missing_count"], 2)
        self.assertEqual(profile["zero_count"], 1)

    def test_spearman_uses_only_pairwise_valid_rows_and_reports_n(self):
        rows = [
            {"a": 1.0, "b": 10.0},
            {"a": 2.0, "b": None},
            {"a": 3.0, "b": 30.0},
            {"a": float("nan"), "b": 40.0},
        ]

        result = spearman_pairwise(rows, "a", "b")

        self.assertEqual(result["valid_pair_count"], 2)
        self.assertTrue(math.isclose(result["spearman_rho"], 1.0))


if __name__ == "__main__":
    unittest.main()
