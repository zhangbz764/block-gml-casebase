"""One-time, restartable repair of the Phase 4B part_06 run_id label."""

from __future__ import annotations

import argparse
from copy import deepcopy
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
from typing import Any


PHASE_DIR = Path(__file__).resolve().parent
OLD_RUN_ID = "full_v032_part_0"
NEW_RUN_ID = "full_v032_part_06"
PART_ID = "part_06"
EXPECTED_ENTRIES = 12620
EXPECTED_COMPLETED = 12618
EXPECTED_EXCLUDED = 2
CSV_NAMES = (
    "encoding_status.csv", "encoding_status_snapshot.csv",
    "building_exclusions.csv", "building_exclusions_snapshot.csv", "chunk_status.csv",
)


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_graph(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def _canonical_without_run_id(record: dict[str, Any]) -> bytes:
    content = deepcopy(record)
    content.get("phase4b", {}).pop("run_id", None)
    return json.dumps(content, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def research_content_sha256(path: Path) -> str:
    return hashlib.sha256(_canonical_without_run_id(_read_graph(path))).hexdigest()


def _write_graph_atomic(path: Path, record: dict[str, Any], expected_content_sha256: str) -> None:
    temporary = path.with_name(path.name + ".tmp.gz")
    if temporary.exists():
        temporary.unlink()
    try:
        with temporary.open("wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
                payload = json.dumps(record, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8") + b"\n"
                compressed.write(payload)
        reloaded = _read_graph(temporary)
        if reloaded.get("phase4b", {}).get("run_id") != NEW_RUN_ID:
            raise RuntimeError(f"temporary Graph did not retain repaired run_id: {path}")
        actual_content_sha256 = hashlib.sha256(_canonical_without_run_id(reloaded)).hexdigest()
        if actual_content_sha256 != expected_content_sha256:
            raise RuntimeError(f"research content changed while repairing {path}")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def _write_csv_atomic(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    os.replace(temporary, path)


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _latest(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    latest: dict[str, dict[str, str]] = {}
    for row in rows:
        sample_id = str(row["sample_id"])
        if sample_id not in latest or int(row.get("attempt") or 1) >= int(latest[sample_id].get("attempt") or 1):
            latest[sample_id] = row
    return list(latest.values())


def _status_counts(rows: list[dict[str, str]]) -> dict[str, int]:
    counts = {"completed": 0, "excluded_no_valid_building": 0, "failed": 0, "interrupted": 0}
    for row in _latest(rows):
        status = str(row.get("status", ""))
        counts[status] = counts.get(status, 0) + 1
    return counts


def _check_label(value: Any, source: str, unknown: list[dict[str, str]]) -> None:
    if value not in (OLD_RUN_ID, NEW_RUN_ID):
        unknown.append({"source": source, "run_id": str(value)})


def repair_part06_run_id(
    phase_dir: Path,
    *,
    apply: bool,
    expected_entries: int = EXPECTED_ENTRIES,
    expected_completed: int = EXPECTED_COMPLETED,
    expected_excluded: int = EXPECTED_EXCLUDED,
) -> dict[str, Any]:
    started = _utc()
    phase_dir = phase_dir.resolve()
    graph_root = phase_dir / "results" / "dataset" / "graphs" / PART_ID
    run_dir = phase_dir / "results" / "runs" / "full_v032" / PART_ID
    audit_dir = phase_dir / "results" / "audit" / "full_v032" / PART_ID
    preflight_path = run_dir / "preflight.json"
    run_summary_path = run_dir / "run_summary.json"
    part_summary_path = audit_dir / "part_summary.json"
    required = [graph_root, preflight_path, run_summary_path, part_summary_path, *[run_dir / name for name in CSV_NAMES]]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"part_06 repair inputs are missing: {missing}")

    graph_paths = sorted(graph_root.rglob("*.graph.json.gz"))
    if len(graph_paths) != expected_completed:
        raise RuntimeError(f"part_06 has {len(graph_paths)} Graphs, expected {expected_completed}")
    unknown: list[dict[str, str]] = []
    graphs_to_modify: list[tuple[Path, dict[str, Any], str]] = []
    graphs_already_correct = 0
    for path in graph_paths:
        record = _read_graph(path)
        metadata = record.get("phase4b", {})
        if metadata.get("part_id") != PART_ID:
            raise RuntimeError(f"Graph has wrong part_id: {path}")
        run_id = metadata.get("run_id")
        _check_label(run_id, str(path), unknown)
        if run_id == OLD_RUN_ID:
            graphs_to_modify.append((path, record, hashlib.sha256(_canonical_without_run_id(record)).hexdigest()))
        elif run_id == NEW_RUN_ID:
            graphs_already_correct += 1

    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if preflight.get("phase4b", {}).get("part_id") != PART_ID:
        raise RuntimeError("preflight.json does not describe part_06")
    _check_label(preflight.get("phase4b", {}).get("run_id"), "preflight.json", unknown)
    run_summary = json.loads(run_summary_path.read_text(encoding="utf-8"))
    if run_summary.get("part_id") != PART_ID:
        raise RuntimeError("run_summary.json does not describe part_06")
    _check_label(run_summary.get("run_id"), "run_summary.json", unknown)

    csv_data: dict[str, tuple[list[str], list[dict[str, str]]]] = {}
    for name in CSV_NAMES:
        fields, rows = _read_csv(run_dir / name)
        if "run_id" not in fields or "part_id" not in fields:
            raise RuntimeError(f"{name} lacks run_id or part_id")
        for row_number, row in enumerate(rows, start=2):
            if row.get("part_id") != PART_ID:
                raise RuntimeError(f"{name}:{row_number} has wrong part_id")
            _check_label(row.get("run_id"), f"{name}:{row_number}", unknown)
        csv_data[name] = fields, rows
    if unknown:
        raise RuntimeError(f"unknown run_id labels found ({len(unknown)}): {unknown[:5]}")

    _, status_rows = csv_data["encoding_status.csv"]
    before_counts = _status_counts(status_rows)
    if len(_latest(status_rows)) != expected_entries:
        raise RuntimeError(f"part_06 has {len(_latest(status_rows))} latest status rows, expected {expected_entries}")
    expected_counts = {"completed": expected_completed, "excluded_no_valid_building": expected_excluded, "failed": 0, "interrupted": 0}
    if before_counts != expected_counts:
        raise RuntimeError(f"part_06 status counts changed: {before_counts}, expected {expected_counts}")

    report: dict[str, Any] = {
        "mode": "apply" if apply else "dry-run", "part_id": PART_ID,
        "old_run_id": OLD_RUN_ID, "new_run_id": NEW_RUN_ID,
        "started_at": started, "graph_count": len(graph_paths),
        "graphs_to_modify": len(graphs_to_modify), "graphs_already_correct": graphs_already_correct,
        "excluded_no_graph": expected_excluded, "unknown_run_id_count": 0,
        "before_status_counts": before_counts, "graphs_modified": 0,
        "research_content_hash_mismatches": 0, "errors": [],
    }
    if not apply:
        report["finished_at"] = _utc()
        return report

    for path, record, content_sha256 in graphs_to_modify:
        record["phase4b"]["run_id"] = NEW_RUN_ID
        _write_graph_atomic(path, record, content_sha256)
        report["graphs_modified"] += 1

    graph_sizes = {path.relative_to(phase_dir).as_posix(): path.stat().st_size for path in graph_paths}
    for name, (fields, rows) in csv_data.items():
        for row in rows:
            row["run_id"] = NEW_RUN_ID
            if name.startswith("encoding_status") and row.get("status") == "completed":
                graph_path = str(row.get("graph_path", ""))
                if graph_path not in graph_sizes:
                    raise RuntimeError(f"status Graph path is outside or missing from part_06: {graph_path}")
                row["output_bytes"] = str(graph_sizes[graph_path])
        _write_csv_atomic(run_dir / name, fields, rows)

    preflight["phase4b"]["run_id"] = NEW_RUN_ID
    _write_json_atomic(preflight_path, preflight)
    _, repaired_status_rows = _read_csv(run_dir / "encoding_status.csv")
    latest = _latest(repaired_status_rows)
    total_output_bytes = sum(int(float(row.get("output_bytes") or 0)) for row in latest)
    run_summary["run_id"] = NEW_RUN_ID
    run_summary["output_bytes"] = total_output_bytes
    _write_json_atomic(run_summary_path, run_summary)
    part_summary = json.loads(part_summary_path.read_text(encoding="utf-8"))
    if part_summary.get("part_id") != PART_ID:
        raise RuntimeError("part_summary.json does not describe part_06")
    part_summary["output_bytes"] = total_output_bytes
    _write_json_atomic(part_summary_path, part_summary)

    after_counts = _status_counts(repaired_status_rows)
    if after_counts != before_counts:
        raise RuntimeError(f"status counts changed during repair: {before_counts} -> {after_counts}")
    report.update({
        "after_status_counts": after_counts, "output_bytes_after": total_output_bytes,
        "graphs_already_correct_after": len(graph_paths), "finished_at": _utc(),
    })
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    try:
        report = repair_part06_run_id(PHASE_DIR, apply=args.apply)
        output = PHASE_DIR / "results" / "audit" / "full_v032" / PART_ID / "run_id_repair_report.json"
        if args.apply:
            _write_json_atomic(output, report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    except Exception as error:
        print(json.dumps({"passed": False, "error_type": type(error).__name__, "message": str(error)}, ensure_ascii=False, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
