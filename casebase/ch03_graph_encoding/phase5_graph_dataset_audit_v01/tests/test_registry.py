from __future__ import annotations

import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


PHASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PHASE_DIR))

from registry import load_audit_spec, load_registry, registry_summary


REGISTRY = PHASE_DIR / "feature_registry_snapshot_v03.tsv"
SPEC = PHASE_DIR / "audit_spec_v01.json"


class RegistryTests(unittest.TestCase):
    def test_registry_is_tab_delimited_and_frozen(self):
        self.assertEqual(pd.read_csv(REGISTRY).shape[1], 1)
        frame = load_registry(REGISTRY)
        self.assertEqual(list(frame.columns), [
            "entity", "field", "role", "unit", "transform_or_encoding",
            "missing_rule", "default_input", "rationale",
        ])
        self.assertEqual(len(frame), 103)
        self.assertEqual(frame.groupby("entity").size().to_dict(), {
            "graph": 21, "block": 26, "node": 26, "cell": 5, "edge": 25,
        })
        self.assertEqual(hashlib.sha256(REGISTRY.read_bytes()).hexdigest(), "88b0ed3b019161c32077672057f8d696b06ef8d1ee480159a0744f874cc7253a")

    def test_registry_roles_core_inputs_and_spec_are_frozen(self):
        frame = load_registry(REGISTRY)
        summary = registry_summary(frame)
        self.assertEqual(summary["role_counts"], {
            "auxiliary_interpretation": 33,
            "core_model": 35,
            "provenance_trace": 21,
            "quality_audit": 14,
        })
        self.assertEqual(summary["default_core_input_count"], 35)
        spec = load_audit_spec(SPEC)
        self.assertEqual(spec["expected_entries"], 213054)
        self.assertEqual(spec["expected_cities"], 24)
        self.assertEqual(spec["phase4b_code_bundle_sha256"], "aec73426e4b9a7ec31d329ada9ed2051971a07fc359eb6907bc4f79c1a9011f0")
        self.assertEqual(spec["part_ids"], [f"part_{index:02d}" for index in range(1, 13)])

    def test_registry_rejects_duplicate_logical_fields(self):
        frame = load_registry(REGISTRY)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "duplicate.tsv"
            pd.concat([frame, frame.iloc[[0]]], ignore_index=True).to_csv(path, sep="\t", index=False)
            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_registry(path, expected_sha256=None)


if __name__ == "__main__":
    unittest.main()
