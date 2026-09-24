from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from fixtures import META, cell_failure_record, completed_status, missing_and_exclusion_record, normal_record, single_node_record
from graph_audit import FeatureAccumulator, audit_graph_record, graph_connectivity


class GraphAuditTests(unittest.TestCase):
    def test_single_node_graph_is_valid(self):
        row, issues = audit_graph_record(single_node_record(), completed_status(), META)
        self.assertEqual(issues, [])
        self.assertEqual((row["connected_components"], row["isolated_node_count"], row["average_degree"], row["edge_density"]), (1, 1, 0.0, 0.0))

    def test_multicomponent_connectivity(self):
        self.assertEqual(graph_connectivity(["A", "B", "C"], [{"source": "A", "target": "B"}]), (2, 1, 2 / 3))

    def test_cell_failure_explains_node_without_a_cell(self):
        row, issues = audit_graph_record(cell_failure_record(), completed_status(), META)
        self.assertEqual(issues, [])
        self.assertEqual((row["node_count"], row["cell_count"], row["cell_failure_count"]), (9, 8, 1))

    def test_cell_failure_coverage_errors_are_distinct(self):
        mutations = {
            "cell_failure_overlap": lambda r: r["cell_failures"][0].update(building_id="N0"),
            "cell_coverage_mismatch": lambda r: r["cells"].pop(),
            "unknown_cell_failure_id": lambda r: r["cell_failures"][0].update(building_id="UNKNOWN"),
            "duplicate_cell_failure_id": lambda r: r["cell_failures"].append(dict(r["cell_failures"][0])),
        }
        for expected, mutate in mutations.items():
            with self.subTest(expected=expected):
                record = cell_failure_record(); mutate(record)
                _, issues = audit_graph_record(record, completed_status(), META)
                self.assertIn(expected, {issue["issue_type"] for issue in issues})

    def test_each_hard_error_is_reported(self):
        mutations = {
            "validation_failed": lambda r: r.update(validation={"passed": False}),
            "metadata_mismatch": lambda r: r["phase4b"].update(code_bundle_sha256="wrong"),
            "duplicate_node_id": lambda r: r["nodes"].append(dict(r["nodes"][0])),
            "duplicate_cell_id": lambda r: r["cells"].append(dict(r["cells"][0])),
            "self_loop": lambda r: r["edges"][0].update(target="A"),
            "duplicate_undirected_edge": lambda r: r["edges"].append({**r["edges"][0], "edge_id": "B--A", "source": "B", "target": "A"}),
            "unknown_edge_endpoint": lambda r: r["edges"][0].update(target="Z"),
            "node_count_mismatch": lambda r: r["graph"].update(node_count=99),
            "building_count_mismatch": lambda r: r["graph"].update(source_building_count=3),
            "excluded_building_leak": lambda r: r.update(exclusions={"buildings": [{"building_id": "A"}]}),
            "weak_edge_count_mismatch": lambda r: r["weak_edge_summary"].update(long_distance_candidate_count=1),
            "nonfinite_numeric": lambda r: r["nodes"][0].update(height_m=math.inf),
        }
        for expected, mutate in mutations.items():
            with self.subTest(expected=expected):
                record = normal_record(); mutate(record)
                _, issues = audit_graph_record(record, completed_status(), META)
                self.assertIn(expected, {issue["issue_type"] for issue in issues})
                self.assertTrue(all({"sample_id", "city", "block_id", "part_id", "issue_type", "message"} <= set(issue) for issue in issues))

    def test_exclusion_ledger_conservation_errors_are_reported(self):
        mutations = {
            "duplicate_exclusion_id": lambda r: r["exclusions"]["buildings"].append(dict(r["exclusions"]["buildings"][0])),
            "excluded_object_count_mismatch": lambda r: r["graph"].update(excluded_building_count=2),
            "exclusion_reason_count_mismatch": lambda r: r["exclusions"]["summary"].update(excluded_building_reason_counts={"other": 1}),
            "exclusion_summary_mismatch": lambda r: r["exclusions"]["summary"].update(encoded_building_count=99),
        }
        for expected, mutate in mutations.items():
            with self.subTest(expected=expected):
                record = missing_and_exclusion_record(); mutate(record)
                _, issues = audit_graph_record(record, completed_status(), META)
                self.assertIn(expected, {issue["issue_type"] for issue in issues})

    def test_streaming_feature_counts_cover_registry(self):
        record = missing_and_exclusion_record()
        row, issues = audit_graph_record(record, completed_status(), META)
        self.assertEqual(issues, [])
        accumulator = FeatureAccumulator()
        accumulator.update(record, row)
        rows = accumulator.to_rows()
        self.assertEqual(len(rows), 103)
        height = next(item for item in rows if item["field"] == "node.height_m")
        self.assertEqual((height["object_count"], height["missing_count"]), (1, 1))
        self.assertFalse(hasattr(accumulator, "records"))
        self.assertEqual(accumulator.issues, [])


if __name__ == "__main__":
    unittest.main()
