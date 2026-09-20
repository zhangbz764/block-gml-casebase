"""Independent consistency check for the frozen phase 3 deliverables."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from export_phase3 import read_graph, validate_v03_record


PHASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PHASE_DIR / "results"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    snapshot = json.loads((RESULTS_DIR / "database_snapshot.json").read_text(encoding="utf-8"))
    manifest_path = RESULTS_DIR / "sample_manifest.csv"
    with (RESULTS_DIR / "audit" / "encoding_status.csv").open(encoding="utf-8-sig", newline="") as handle:
        status_rows = list(csv.DictReader(handle))
    completed = [row for row in status_rows if row["status"] == "completed"]
    failed = [row for row in status_rows if row["status"] == "failed"]
    errors: list[str] = []

    if sha256_file(manifest_path) != snapshot["sample_manifest_sha256"]:
        errors.append("sample manifest hash mismatch")
    if len(status_rows) != snapshot["selected_unique_count"]:
        errors.append("status row count does not match frozen manifest")
    if any(row["status"] not in {"completed", "failed", "excluded_with_reason"} for row in status_rows):
        errors.append("unexpected status value")
    if any(not row["failure_message"] for row in failed):
        errors.append("failed sample lacks failure_message")

    graph_dir = RESULTS_DIR / "dataset" / "graphs"
    actual_files = {path.name.removesuffix(".graph.json.gz") for path in graph_dir.glob("*.graph.json.gz")}
    expected_files = {row["sample_id"] for row in completed}
    if actual_files != expected_files:
        errors.append("graph file set does not exactly match completed statuses")

    city_reloads: Counter[str] = Counter()
    for status in completed:
        path = graph_dir / f"{status['sample_id']}.graph.json.gz"
        record = read_graph(path)
        city_reloads[status["city"]] += 1
        if record.get("sample_manifest_sha256") != snapshot["sample_manifest_sha256"]:
            errors.append(f"{status['sample_id']}: embedded manifest hash mismatch")
        validation = validate_v03_record(record)
        if not validation["passed"]:
            errors.append(f"{status['sample_id']}: {validation['issue_types']}")
    if len(city_reloads) != 24 or any(count < 2 for count in city_reloads.values()):
        errors.append("fewer than two successful graph reloads for at least one city")

    summary = {
        "planned_samples": len(status_rows),
        "completed_samples": len(completed),
        "failed_samples": len(failed),
        "graph_files": len(actual_files),
        "cities_with_at_least_two_successful_reloads": sum(count >= 2 for count in city_reloads.values()),
        "all_completed_graphs_reloaded": len(completed),
        "sample_manifest_sha256": snapshot["sample_manifest_sha256"],
        "verification_passed": not errors,
        "errors": errors,
    }
    (RESULTS_DIR / "verification_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
