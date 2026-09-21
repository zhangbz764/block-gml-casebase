"""Verify the complete Phase 3.1 evidence contract without database access."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PHASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PHASE_DIR / "results"
CH03_DIR = PHASE_DIR.parent
PHASE3_DIR = CH03_DIR / "phase3_encoding_dataset_v03"
sys.path.insert(0, str(PHASE3_DIR))

from export_phase3 import read_graph


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _truth(value: str) -> bool:
    return value.strip().lower() == "true"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_inventory() -> None:
    inventory_path = RESULTS_DIR / "output_inventory.csv"
    paths = sorted(
        path for path in RESULTS_DIR.rglob("*")
        if path.is_file() and path != inventory_path
    )
    with inventory_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("relative_path", "type", "size_bytes", "sha256", "modified_at"),
        )
        writer.writeheader()
        for path in paths:
            stat = path.stat()
            writer.writerow(
                {
                    "relative_path": path.relative_to(RESULTS_DIR).as_posix(),
                    "type": "graph_json_gzip" if path.name.endswith(".graph.json.gz") else path.suffix.lstrip("."),
                    "size_bytes": stat.st_size,
                    "sha256": _sha256(path),
                    "modified_at": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
                }
            )


def verify() -> dict[str, Any]:
    affected = _read_csv(RESULTS_DIR / "affected_manifest.csv")
    summary = _read_csv(RESULTS_DIR / "audit" / "revised_graph_summary.csv")
    exclusions = _read_csv(RESULTS_DIR / "audit" / "excluded_buildings.csv")
    regressions = _read_csv(RESULTS_DIR / "audit" / "regression_signature_check.csv")
    old_status = _read_csv(PHASE3_DIR / "results" / "audit" / "encoding_status.csv")
    issues: list[dict[str, str]] = []

    def add(check: str, detail: str) -> None:
        issues.append({"check": check, "detail": detail})

    if len(affected) != 57:
        add("affected_count", f"expected 57, got {len(affected)}")
    if len(summary) != 57 or len({row["sample_id"] for row in summary}) != 57:
        add("status_coverage", f"expected 57 unique rows, got {len(summary)}")
    if len(exclusions) != 308:
        add("excluded_count", f"expected 308, got {len(exclusions)}")
    exclusion_keys = [(row["sample_id"], row["building_id"]) for row in exclusions]
    if len(exclusion_keys) != len(set(exclusion_keys)):
        add("excluded_unique", "duplicate sample/building exclusion records exist")
    if any(row["reason"] != "empty_after_shrink" for row in exclusions):
        add("excluded_reason", "an exclusion has an unexpected reason")

    exclusions_by_sample: dict[str, set[str]] = {}
    for row in exclusions:
        exclusions_by_sample.setdefault(row["sample_id"], set()).add(row["building_id"])

    completed = [row for row in summary if row["status"] == "completed"]
    all_excluded = [row for row in summary if row["status"] == "excluded_no_valid_building"]
    failed = [row for row in summary if row["status"] == "failed"]
    if len(completed) != 56:
        add("completed_count", f"expected 56, got {len(completed)}")
    if [row["sample_id"] for row in all_excluded] != ["p3__CH_ZU_000062"]:
        add("all_excluded_case", f"unexpected cases: {[row['sample_id'] for row in all_excluded]}")
    if failed:
        add("execution_failures", "; ".join(f"{row['sample_id']}: {row['failure_message']}" for row in failed))

    for row in completed:
        sample_id = row["sample_id"]
        graph_path = CH03_DIR / row["graph_path"]
        if not graph_path.exists():
            add("graph_exists", sample_id)
            continue
        record = read_graph(graph_path)
        graph = record["graph"]
        if graph["node_count"] != graph["cell_count"] or graph["cell_failure_count"] != 0:
            add("graph_counts", sample_id)
        if not record.get("validation", {}).get("passed"):
            add("graph_validation", sample_id)
        excluded_ids = exclusions_by_sample.get(sample_id, set())
        node_ids = {str(item["building_id"]) for item in record["nodes"]}
        cell_ids = {str(item["building_id"]) for item in record["cells"]}
        edge_ids = {
            str(endpoint)
            for edge in record["edges"]
            for endpoint in (edge["source"], edge["target"])
        }
        leaked = excluded_ids & (node_ids | cell_ids | edge_ids)
        if leaked:
            add("excluded_id_leak", f"{sample_id}: {sorted(leaked)}")
        recorded_ids = {
            str(item["building_id"])
            for item in record.get("exclusions", {}).get("buildings", [])
        }
        if recorded_ids != excluded_ids:
            add("embedded_exclusion_ids", sample_id)
        if not _truth(row["roundtrip_ok"]):
            add("roundtrip", sample_id)

        baseline_row = next(item for item in affected if item["sample_id"] == sample_id)
        old_graph = read_graph(CH03_DIR / baseline_row["baseline_graph_path"])
        old_failure_ids = {
            str(item["building_id"])
            for item in old_graph["cell_failures"]
            if item.get("failure_type") == "empty_after_shrink"
        }
        if old_failure_ids != excluded_ids:
            add("baseline_exclusion_match", sample_id)

    excluded_graph_path = RESULTS_DIR / "dataset" / "graphs" / "p3__CH_ZU_000062.graph.json.gz"
    if excluded_graph_path.exists():
        add("no_empty_graph", str(excluded_graph_path))
    old_all_excluded = next(item for item in affected if item["sample_id"] == "p3__CH_ZU_000062")
    old_all_record = read_graph(CH03_DIR / old_all_excluded["baseline_graph_path"])
    old_all_ids = {str(item["building_id"]) for item in old_all_record["cell_failures"]}
    if old_all_ids != exclusions_by_sample.get("p3__CH_ZU_000062", set()):
        add("all_excluded_baseline_match", "p3__CH_ZU_000062")

    if len(regressions) != 3:
        add("regression_count", f"expected 3, got {len(regressions)}")
    for row in regressions:
        if row["status"] != "completed" or not _truth(row["all_match"]):
            add("regression_signature", f"{row['sample_id']}: {row['failure_message']}")
        if int(row.get("excluded_building_count") or -1) != 0:
            add("regression_exclusion", row["sample_id"])

    invalid_failures = [
        row for row in old_status
        if row["status"] == "failed" and row["failure_message"].startswith("Invalid footprint for building")
    ]
    if len(invalid_failures) != 14:
        add("unchanged_invalid_footprint", f"expected 14 baseline failures, got {len(invalid_failures)}")

    result = {
        "experiment_id": "CH3-P3.1-NODE-CLEANUP-V031",
        "status": "VERIFIED" if not issues else "FAILED",
        "passed": not issues,
        "affected_block_count": len(summary),
        "excluded_building_count": len(exclusions),
        "completed_graph_count": len(completed),
        "excluded_graph_count": len(all_excluded),
        "execution_failure_count": len(failed),
        "regression_case_count": len(regressions),
        "regression_match_count": sum(_truth(row.get("all_match", "")) for row in regressions),
        "unchanged_invalid_footprint_failure_count": len(invalid_failures),
        "issue_count": len(issues),
        "issues": issues,
    }
    (RESULTS_DIR / "verification_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_inventory()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    raise SystemExit(0 if verify()["passed"] else 1)
