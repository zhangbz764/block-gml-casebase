from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from merge_graph_audit import merge_feature_counts, merge_part_indexes, write_dataset_summaries
from write_researcher_summary import build_researcher_summary


def row(sample: str, block: str, status: str = "completed") -> dict:
    return {"sample_id": sample, "city": "City", "block_id": block, "part_id": "part_01", "case_status": status,
            "audit_issue_count": 0, "node_count": 2 if status == "completed" else None, "edge_count": 1 if status == "completed" else None,
            "cell_failure_count": 0 if status == "completed" else None,
            "average_degree": 1, "edge_density": 1, "connected_components": 1, "isolated_node_ratio": 0,
            "long_distance_candidate_ratio": 0, "low_shared_candidate_ratio": 0, "combined_weak_candidate_ratio": 0,
            "excluded_building_ratio": 0 if status == "completed" else 1, "excluded_building_count": 0 if status == "completed" else 1}


class MergeTests(unittest.TestCase):
    def test_researcher_summary_reports_counts_tail_cities_and_four_questions(self):
        global_summary = {"entry_count": 213054, "completed_graph_count": 213000,
                          "excluded_no_valid_building_count": 54, "hard_issue_entry_count": 0,
                          "node_total": 900000, "edge_total": 1800000, "cell_failure_total": 12,
                          "excluded_building_total": 727}
        cities = pd.DataFrame([
            {"city": "City A", "node_count_p99": 100, "edge_count_p99": 200, "connected_components_p99": 3,
             "isolated_node_ratio_p99": 0.1, "combined_weak_candidate_ratio_p99": 0.2, "excluded_building_ratio_p99": 0.01},
            {"city": "City B", "node_count_p99": 80, "edge_count_p99": 220, "connected_components_p99": 2,
             "isolated_node_ratio_p99": 0.2, "combined_weak_candidate_ratio_p99": 0.1, "excluded_building_ratio_p99": 0.03},
        ])
        text = build_researcher_summary(global_summary, cities, {"automated_audit_passed": True}, expected_entries=213054)
        for value in ("213,054", "213,000", "54", "900,000", "1,800,000", "12", "727", "City A", "City B"):
            self.assertIn(value, text)
        self.assertIn("自动一致性检查通过", text)
        self.assertEqual(text.count("需要研究者判断"), 1)
        self.assertEqual(sum(line.startswith(("1.", "2.", "3.", "4.")) for line in text.splitlines()), 4)
        self.assertLess(len(text), 5000)

    def test_merge_rejects_wrong_count_and_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "p.csv.gz"
            pd.DataFrame([row("S", "B"), row("S", "B2")]).to_csv(path, index=False, compression="gzip")
            with self.assertRaisesRegex(ValueError, "expected 3"): merge_part_indexes([path], 3)
            with self.assertRaisesRegex(ValueError, "duplicate sample_id"): merge_part_indexes([path], 2)

    def test_summary_uses_only_completed_graphs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); path = root / "p.csv.gz"
            pd.DataFrame([row("S1", "B1"), row("S2", "B2", "excluded_no_valid_building")]).to_csv(path, index=False, compression="gzip")
            index = merge_part_indexes([path], 2)
            features = pd.DataFrame([{"scope": "__global__", "entity": "graph", "field": "graph.sample_id", "role": "provenance_trace", "unit": "id", "transform_or_encoding": "none", "missing_rule": "not allowed", "default_input": "no", "rationale": "x", "object_count": 1, "observed_count": 1, "missing_count": 0, "nonfinite_count": 0, "numeric_count": 0, "numeric_sum": 0, "numeric_sum_squares": 0, "numeric_min": None, "numeric_max": None}])
            summary = write_dataset_summaries(index, features, root / "audit")
            self.assertEqual((summary["entry_count"], summary["completed_graph_count"], summary["excluded_no_valid_building_count"]), (2, 1, 1))

    def test_feature_counts_sum(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for index in range(2):
                path = Path(tmp) / f"f{index}.csv"
                pd.DataFrame([{"scope": "__global__", "entity": "node", "field": "node.height_m", "role": "auxiliary_interpretation", "unit": "m", "transform_or_encoding": "none", "missing_rule": "retain null and mask", "default_input": "no", "rationale": "x", "object_count": 2, "observed_count": 1, "missing_count": 1, "nonfinite_count": 0, "numeric_count": 1, "numeric_sum": 10, "numeric_sum_squares": 100, "numeric_min": 10, "numeric_max": 10}]).to_csv(path, index=False)
                paths.append(path)
            merged = merge_feature_counts(paths)
            self.assertEqual((merged.iloc[0]["object_count"], merged.iloc[0]["missing_count"], merged.iloc[0]["missing_rate"]), (4, 2, 0.5))


if __name__ == "__main__": unittest.main()
