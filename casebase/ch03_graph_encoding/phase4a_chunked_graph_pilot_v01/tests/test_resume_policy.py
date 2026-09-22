from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


PHASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PHASE_DIR))

from run_phase4a import atomic_write_graph, resume_action


EXPECTED = {
    "manifest_sha256": "manifest-hash",
    "encoding_version": "0.3.1",
    "run_id": "smoke_v01",
    "parameter_sha256": "parameter-hash",
}


class ResumePolicyTests(unittest.TestCase):
    def test_atomic_graph_roundtrip_and_completed_resume_skip(self):
        record = {
            "nodes": [], "cells": [], "edges": [],
            "phase4a": EXPECTED,
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "case.graph.json.gz"
            atomic_write_graph(record, path)
            action, attempt = resume_action(
                {"status": "completed", "attempt": 1},
                path,
                EXPECTED,
                retry_failed=False,
            )

        self.assertEqual((action, attempt), ("skip", 1))

    def test_mismatched_completed_graph_stops_and_failed_requires_retry_flag(self):
        record = {"nodes": [], "cells": [], "edges": [], "phase4a": {**EXPECTED, "run_id": "old"}}
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "case.graph.json.gz"
            atomic_write_graph(record, path)
            with self.assertRaises(RuntimeError):
                resume_action({"status": "completed", "attempt": 1}, path, EXPECTED, False)
            self.assertEqual(
                resume_action({"status": "failed", "attempt": 2}, path.with_name("missing.gz"), EXPECTED, False),
                ("skip_failed", 2),
            )
            self.assertEqual(
                resume_action({"status": "failed", "attempt": 2}, path.with_name("missing.gz"), EXPECTED, True),
                ("run", 3),
            )


if __name__ == "__main__":
    unittest.main()
