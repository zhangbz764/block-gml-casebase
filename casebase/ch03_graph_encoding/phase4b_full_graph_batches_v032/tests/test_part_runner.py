from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


PHASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PHASE_DIR))

from run_phase4b import PART_IDS, atomic_write_graph, build_parser, resume_action, validate_part_rows


META = {
    "source_manifest_sha256": "source", "full_manifest_sha256": "full",
    "part_manifest_sha256": "part", "encoding_spec_sha256": "spec",
    "encoding_version": "0.3.2", "part_id": "part_01", "run_id": "run",
    "parameter_sha256": "parameters",
}


class PartRunnerTests(unittest.TestCase):
    def test_cli_requires_one_valid_part_and_has_no_all_option(self):
        parser = build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args([])
        with self.assertRaises(SystemExit):
            parser.parse_args(["--part-id", "part_13", "--run-id", "run"])
        args = parser.parse_args(["--part-id", "part_01", "--run-id", "run"])
        self.assertIn(args.part_id, PART_IDS)
        self.assertNotIn("all", vars(args))

    def test_cross_part_rows_are_rejected(self):
        with self.assertRaises(RuntimeError):
            validate_part_rows([{"part_id": "part_02"}], "part_01")

    def test_resume_accepts_only_matching_completed_graph_and_retries_failed_explicitly(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "graph.json.gz"
            atomic_write_graph({"phase4b": META, "nodes": [], "cells": [], "edges": [], "validation": {"passed": True}}, path)
            self.assertEqual(resume_action({"status": "completed", "attempt": 1}, path, META, False), ("skip", 1))
            self.assertEqual(resume_action({"status": "failed", "attempt": 2}, path.with_name("missing.gz"), META, True), ("run", 3))
            with self.assertRaises(RuntimeError):
                resume_action({"status": "completed", "attempt": 1}, path, {**META, "part_id": "part_02"}, False)


if __name__ == "__main__":
    unittest.main()
