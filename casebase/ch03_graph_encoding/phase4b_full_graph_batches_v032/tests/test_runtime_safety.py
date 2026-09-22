from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


PHASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PHASE_DIR))

import run_phase4b as runner
import verify_phase4b as verifier


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class RuntimeSafetyTests(unittest.TestCase):
    def test_case_ledgers_append_and_chunk_snapshot_is_atomic(self):
        append_row = getattr(runner, "append_csv_row", None)
        snapshot = getattr(runner, "snapshot_csv_ledger", None)
        self.assertIsNotNone(append_row)
        self.assertIsNotNone(snapshot)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ledger = root / "encoding_status.csv"
            frozen = root / "encoding_status_snapshot.csv"
            fields = ["sample_id", "status"]
            append_row(ledger, {"sample_id": "a", "status": "completed"}, fields)
            append_row(ledger, {"sample_id": "b", "status": "failed"}, fields)
            snapshot(ledger, frozen, fields)

            self.assertEqual(pd.read_csv(ledger).to_dict("records"), [
                {"sample_id": "a", "status": "completed"},
                {"sample_id": "b", "status": "failed"},
            ])
            self.assertEqual(pd.read_csv(frozen).to_dict("records"), pd.read_csv(ledger).to_dict("records"))
            self.assertFalse((root / "encoding_status_snapshot.csv.tmp").exists())

            append_row(ledger, {"sample_id": "c", "status": "completed"}, fields)
            self.assertEqual(len(pd.read_csv(ledger)), 3)
            self.assertEqual(len(pd.read_csv(frozen)), 2)

    def test_source_full_and_part_manifest_hashes_are_all_enforced(self):
        verify = getattr(runner, "verify_frozen_manifest_files", None)
        self.assertIsNotNone(verify)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.gz"; source.write_bytes(b"source")
            full = root / "full.gz"; full.write_bytes(b"full")
            part = root / "part.gz"; part.write_bytes(b"part")
            recorded = {
                "source_manifest_sha256": digest(source),
                "full_run_manifest_v032.csv.gz": digest(full),
                "parts": {"part_01": digest(part)},
            }
            verify(source, full, part, "part_01", recorded)
            source.write_bytes(b"changed")
            with self.assertRaises(RuntimeError):
                verify(source, full, part, "part_01", recorded)

    def test_code_bundle_detects_changes_and_existing_parts_must_match(self):
        module_spec = importlib.util.find_spec("code_integrity")
        self.assertIsNotNone(module_spec)
        integrity = importlib.import_module("code_integrity")
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first = root / "first.py"; first.write_text("a = 1\n", encoding="utf-8")
            second = root / "second.py"; second.write_text("b = 2\n", encoding="utf-8")
            frozen = integrity.compute_code_manifest([first, second], root)
            integrity.verify_code_manifest([first, second], root, frozen)
            second.write_text("b = 3\n", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                integrity.verify_code_manifest([first, second], root, frozen)

            run_root = root / "runs"
            preflight = run_root / "part_01" / "preflight.json"
            preflight.parent.mkdir(parents=True)
            preflight.write_text(json.dumps({"phase4b": {"code_bundle_sha256": "code-v1"}}), encoding="utf-8")
            runner.verify_existing_part_code_hashes(run_root, "code-v1")
            with self.assertRaises(RuntimeError):
                runner.verify_existing_part_code_hashes(run_root, "code-v2")

    def test_building_count_mismatch_is_a_case_failure_condition(self):
        check = getattr(runner, "require_matching_building_count", None)
        error_type = getattr(runner, "BuildingCountMismatch", None)
        self.assertIsNotNone(check)
        self.assertIsNotNone(error_type)
        check(2, [{"building_id": "a"}, {"building_id": "b"}])
        with self.assertRaises(error_type):
            check(2, [{"building_id": "a"}])

    def test_failed_or_interrupted_summary_returns_nonzero(self):
        exit_code = getattr(runner, "summary_exit_code", None)
        self.assertIsNotNone(exit_code)
        self.assertEqual(exit_code({"failed": 0, "interrupted": 0}), 0)
        self.assertEqual(exit_code({"failed": 1, "interrupted": 0}), 1)
        self.assertEqual(exit_code({"failed": 0, "interrupted": 1}), 1)

    def test_completed_graph_requires_v032_validation_part_and_code_hash(self):
        problems = getattr(verifier, "completed_graph_problems", None)
        self.assertIsNotNone(problems)
        expected = {
            "encoding_version": "0.3.2",
            "part_id": "part_01",
            "source_manifest_sha256": "source",
            "full_manifest_sha256": "full",
            "part_manifest_sha256": "part",
            "encoding_spec_sha256": "spec",
            "parameter_sha256": "parameters",
            "code_bundle_sha256": "code-v1",
        }
        record = {
            "encoding_version": "0.3.2",
            "validation": {"passed": True},
            "phase4b": dict(expected),
        }
        self.assertEqual(problems(record, expected), [])
        broken = {**record, "validation": {"passed": False}, "phase4b": {**expected, "code_bundle_sha256": "code-v2"}}
        actual = problems(broken, expected)
        self.assertTrue(any("validation" in item for item in actual))
        self.assertTrue(any("code_bundle_sha256" in item for item in actual))


if __name__ == "__main__":
    unittest.main()
