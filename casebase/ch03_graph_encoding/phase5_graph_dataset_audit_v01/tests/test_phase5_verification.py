from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from registry import load_registry
from run_phase5 import build_parser, ensure_manual_review_matches
from verify_phase5 import manual_review_status, validate_part_audit_meta, validate_phase5_tables


class VerificationTests(unittest.TestCase):
    def test_part_meta_rejects_old_phase5_code_bundle(self):
        matching = {"passed": True, "phase5_code_bundle_sha256": "current"}
        self.assertEqual(validate_part_audit_meta(matching, "current", "part_01"), [])
        issues = validate_part_audit_meta(matching, "changed", "part_01")
        self.assertTrue(any("Phase 5 code bundle" in issue for issue in issues))

    def test_manual_review_rejects_same_length_different_samples_without_overwrite(self):
        manifest = pd.DataFrame({"sample_id": ["S1", "S2"]})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manual_review.csv"
            original = pd.DataFrame({"sample_id": ["S1", "OLD"], "visual_status": ["pass", "pass"],
                                     "issue_type": ["", ""], "comment": ["kept", "kept"]})
            original.to_csv(path, index=False, encoding="utf-8-sig")
            before = path.read_bytes()
            with self.assertRaisesRegex(RuntimeError, "rename"):
                ensure_manual_review_matches(manifest, path)
            self.assertEqual(path.read_bytes(), before)

    def test_manual_review_completion_requires_same_ids_order_and_filled_status(self):
        review = pd.DataFrame({"sample_id": ["S1", "S2"]})
        mismatched = pd.DataFrame({"sample_id": ["S1", "OLD"], "visual_status": ["pass", "pass"]})
        status, issues = manual_review_status(review, mismatched, 2)
        self.assertEqual(status, "MANIFEST_MISMATCH"); self.assertTrue(issues)
        reordered = pd.DataFrame({"sample_id": ["S2", "S1"], "visual_status": ["pass", "pass"]})
        status, issues = manual_review_status(review, reordered, 2)
        self.assertEqual(status, "MANIFEST_MISMATCH"); self.assertTrue(any("order" in item for item in issues))
        pending = pd.DataFrame({"sample_id": ["S1", "S2"], "visual_status": ["pass", ""]})
        self.assertEqual(manual_review_status(review, pending, 2), ("PENDING", []))
        complete = pd.DataFrame({"sample_id": ["S1", "S2"], "visual_status": ["pass", "pass"]})
        self.assertEqual(manual_review_status(review, complete, 2), ("COMPLETED", []))

    def test_cli_has_no_default_formal_mode(self):
        with self.assertRaises(SystemExit): build_parser().parse_args([])
        args = build_parser().parse_args(["--part-id", "part_01", "--run-id", "full_v01"])
        self.assertEqual(args.part_id, "part_01")

    def test_compact_tables_pass_and_detect_failed_status(self):
        registry = load_registry(HERE.parent / "feature_registry_snapshot_v03.tsv")
        cities = [f"City {index:02d}" for index in range(24)]
        index_rows = []
        review_rows = []
        for city in cities:
            for case in range(2):
                item = {"sample_id": f"{city}-{case}", "city": city, "block_id": f"B{case}", "case_status": "completed",
                        "graph_exists": True, "graph_readable": True, "validation_passed": True, "audit_passed": True, "graph_path": "graphs/x"}
                index_rows.append(item); review_rows.append({**item, "selection_source": "city_median" if case == 0 else "city_structural_extreme"})
        features = []
        for scope in ["__global__", *cities]:
            for row in registry.to_dict("records"): features.append({"scope": scope, **row})
        spec = {"expected_entries": 48, "expected_cities": 24, "expected_city_names": cities,
                "registry_entity_counts": {"graph": 21, "block": 26, "node": 26, "cell": 5, "edge": 25},
                "registry_role_counts": {"core_model": 35, "auxiliary_interpretation": 33, "provenance_trace": 21, "quality_audit": 14},
                "review_case_count": 48, "review_cases_per_city": 2}
        index = pd.DataFrame(index_rows)
        self.assertEqual(validate_phase5_tables(index, pd.DataFrame(features), pd.DataFrame(review_rows), spec), [])
        index.loc[0, "case_status"] = "failed"
        self.assertTrue(any("failed/interrupted/missing" in issue for issue in validate_phase5_tables(index, pd.DataFrame(features), pd.DataFrame(review_rows), spec)))


if __name__ == "__main__": unittest.main()
