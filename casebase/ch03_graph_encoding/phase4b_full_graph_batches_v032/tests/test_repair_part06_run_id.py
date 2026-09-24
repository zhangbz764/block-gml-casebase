from __future__ import annotations

import csv
import gzip
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest


PHASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PHASE_DIR))

from repair_part06_run_id import (
    NEW_RUN_ID,
    OLD_RUN_ID,
    repair_part06_run_id,
    research_content_sha256,
)


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_graph(path: Path, run_id: str, sample_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "encoding_version": "0.3.2",
        "sample": {"sample_id": sample_id},
        "phase4b": {"part_id": "part_06", "run_id": run_id, "code_bundle_sha256": "code"},
        "nodes": [{"building_id": sample_id + "_B"}], "cells": [], "edges": [],
        "validation": {"passed": True},
    }
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as handle:
            handle.write((json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8"))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def fixture(root: Path, graph_ids: tuple[str, str] = (OLD_RUN_ID, OLD_RUN_ID)) -> tuple[Path, list[Path]]:
    graphs = [
        root / "results/dataset/graphs/part_06/City/chunk_0001/S1.graph.json.gz",
        root / "results/dataset/graphs/part_06/City/chunk_0001/S2.graph.json.gz",
    ]
    for path, run_id, sample in zip(graphs, graph_ids, ("S1", "S2")):
        write_graph(path, run_id, sample)
    run = root / "results/runs/full_v032/part_06"; run.mkdir(parents=True)
    (run / "preflight.json").write_text(json.dumps({"phase4b": {"part_id": "part_06", "run_id": OLD_RUN_ID}}), encoding="utf-8")
    summary = {"run_id": OLD_RUN_ID, "part_id": "part_06", "planned": 3, "completed": 2,
               "excluded_no_valid_building": 1, "failed": 0, "interrupted": 0, "output_bytes": sum(path.stat().st_size for path in graphs)}
    (run / "run_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    status_fields = ["run_id", "part_id", "attempt", "sample_id", "status", "output_bytes", "graph_path"]
    status_rows = [
        {"run_id": OLD_RUN_ID, "part_id": "part_06", "attempt": 1, "sample_id": "S1", "status": "completed", "output_bytes": graphs[0].stat().st_size, "graph_path": graphs[0].relative_to(root).as_posix()},
        {"run_id": OLD_RUN_ID, "part_id": "part_06", "attempt": 1, "sample_id": "S2", "status": "completed", "output_bytes": graphs[1].stat().st_size, "graph_path": graphs[1].relative_to(root).as_posix()},
        {"run_id": OLD_RUN_ID, "part_id": "part_06", "attempt": 1, "sample_id": "S3", "status": "excluded_no_valid_building", "output_bytes": 0, "graph_path": ""},
    ]
    for name in ("encoding_status.csv", "encoding_status_snapshot.csv"):
        write_csv(run / name, status_rows, status_fields)
    exclusion_fields = ["run_id", "part_id", "sample_id", "building_id", "reason"]
    exclusion_rows = [{"run_id": OLD_RUN_ID, "part_id": "part_06", "sample_id": "S3", "building_id": "B3", "reason": "empty_after_shrink"}]
    for name in ("building_exclusions.csv", "building_exclusions_snapshot.csv"):
        write_csv(run / name, exclusion_rows, exclusion_fields)
    write_csv(run / "chunk_status.csv", [{"run_id": OLD_RUN_ID, "part_id": "part_06", "status": "completed"}], ["run_id", "part_id", "status"])
    (run / "run_log.txt").write_text(f"command --run-id {OLD_RUN_ID}\n", encoding="utf-8")
    audit = root / "results/audit/full_v032/part_06"; audit.mkdir(parents=True)
    (audit / "part_summary.json").write_text(json.dumps({key: value for key, value in summary.items() if key != "run_id"}), encoding="utf-8")
    return run, graphs


class RepairPart06RunIdTests(unittest.TestCase):
    def test_dry_run_does_not_write_and_apply_is_idempotent(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); run, graphs = fixture(root)
            tracked = [*graphs, *run.glob("*"), root / "results/audit/full_v032/part_06/part_summary.json"]
            before = {path: file_hash(path) for path in tracked}
            report = repair_part06_run_id(root, apply=False, expected_entries=3, expected_completed=2, expected_excluded=1)
            self.assertEqual((report["graphs_to_modify"], report["excluded_no_graph"], report["unknown_run_id_count"]), (2, 1, 0))
            self.assertEqual(before, {path: file_hash(path) for path in tracked})

            content_before = {path: research_content_sha256(path) for path in graphs}
            applied = repair_part06_run_id(root, apply=True, expected_entries=3, expected_completed=2, expected_excluded=1)
            self.assertEqual(applied["graphs_modified"], 2)
            self.assertEqual(content_before, {path: research_content_sha256(path) for path in graphs})
            for path in graphs:
                with gzip.open(path, "rt", encoding="utf-8") as handle:
                    self.assertEqual(json.load(handle)["phase4b"]["run_id"], NEW_RUN_ID)
            with (run / "encoding_status.csv").open(encoding="utf-8-sig") as handle:
                status = list(csv.DictReader(handle))
            self.assertEqual({row["run_id"] for row in status}, {NEW_RUN_ID})
            self.assertEqual([int(row["output_bytes"]) for row in status[:2]], [path.stat().st_size for path in graphs])
            self.assertEqual((run / "run_log.txt").read_text(encoding="utf-8"), f"command --run-id {OLD_RUN_ID}\n")
            again = repair_part06_run_id(root, apply=True, expected_entries=3, expected_completed=2, expected_excluded=1)
            self.assertEqual((again["graphs_modified"], again["graphs_already_correct"]), (0, 2))

    def test_unknown_label_refuses_before_any_write(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); run, graphs = fixture(root, (OLD_RUN_ID, "unexpected"))
            before = {path: file_hash(path) for path in [*graphs, run / "preflight.json"]}
            with self.assertRaisesRegex(RuntimeError, "unknown run_id"):
                repair_part06_run_id(root, apply=True, expected_entries=3, expected_completed=2, expected_excluded=1)
            self.assertEqual(before, {path: file_hash(path) for path in before})

    def test_partially_repaired_graphs_resume_to_completion(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); _, graphs = fixture(root, (NEW_RUN_ID, OLD_RUN_ID))
            report = repair_part06_run_id(root, apply=True, expected_entries=3, expected_completed=2, expected_excluded=1)
            self.assertEqual((report["graphs_modified"], report["graphs_already_correct"]), (1, 1))
            for path in graphs:
                with gzip.open(path, "rt", encoding="utf-8") as handle:
                    self.assertEqual(json.load(handle)["phase4b"]["run_id"], NEW_RUN_ID)


if __name__ == "__main__":
    unittest.main()
