"""Run recoverable Phase 4A graph encoding in city-scoped chunks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time
import tracemalloc
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


PHASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PHASE_DIR / "results"
CH03_DIR = PHASE_DIR.parent
PHASE31_DIR = CH03_DIR / "phase3_1_encoding_cleanup_v031"
PHASE3_DIR = CH03_DIR / "phase3_encoding_dataset_v03"
sys.path.insert(0, str(CH03_DIR))
sys.path.insert(0, str(PHASE31_DIR))
sys.path.insert(0, str(PHASE3_DIR))

from batch_db_reader import (
    estimate_transfer_bytes,
    fetch_block_batch,
    fetch_building_batch,
    iter_city_chunks,
)
from export_phase3 import DEFAULT_TESSELLATION_CONFIG, read_graph, write_graph
from export_phase3_1 import ENCODING_VERSION, encode_block_to_v031
from shared.db_readonly import readonly_connection


SPEC = json.loads((PHASE_DIR / "pilot_spec_v0.1.json").read_text(encoding="utf-8"))
SOURCE_MANIFEST_SHA256 = SPEC["source_manifest_sha256"]
STATUS_FIELDS = [
    "run_id", "attempt", "sample_id", "city", "block_id", "city_count_stratum",
    "sample_weight", "chunk_id", "status", "planned_building_count",
    "actual_building_count", "building_count_matches", "query_seconds",
    "encoding_seconds", "runtime_seconds", "node_count", "edge_count", "cell_count",
    "excluded_building_count", "output_bytes", "python_peak_memory_mb", "graph_path",
    "failure_type", "failure_message",
]
CHUNK_FIELDS = [
    "run_id", "city", "chunk_id", "planned_blocks", "returned_blocks",
    "returned_buildings", "block_query_seconds", "building_query_seconds",
    "query_seconds", "estimated_transfer_bytes", "status", "failure_message",
]
EXCLUSION_FIELDS = [
    "run_id", "attempt", "sample_id", "city", "block_id", "building_id", "reason",
    "footprint_area_m2", "footprint_perimeter_m", "footprint_compactness",
    "footprint_covered_by_block",
]


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _parameter_snapshot(chunk_size: int) -> dict[str, Any]:
    return {
        "chunk_size": int(chunk_size),
        "tessellation": dict(DEFAULT_TESSELLATION_CONFIG.__dict__),
    }


def _expected_metadata(run_id: str, chunk_size: int) -> dict[str, Any]:
    parameters = _parameter_snapshot(chunk_size)
    return {
        "manifest_sha256": SOURCE_MANIFEST_SHA256,
        "encoding_version": ENCODING_VERSION,
        "run_id": run_id,
        "parameter_sha256": _sha256_bytes(
            json.dumps(parameters, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ),
        "parameters": parameters,
    }


def atomic_write_graph(record: dict[str, Any], path: Path) -> None:
    """Write, read back, and atomically publish one compressed graph."""
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite graph: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.gz")
    if temporary.exists():
        temporary.unlink()
    try:
        write_graph(record, temporary)
        reloaded = read_graph(temporary)
        if reloaded.get("phase4a") != record.get("phase4a"):
            raise RuntimeError("Graph metadata changed during compression roundtrip")
        for field in ("nodes", "cells", "edges"):
            if len(reloaded.get(field, [])) != len(record.get(field, [])):
                raise RuntimeError(f"Graph {field} count changed during roundtrip")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def resume_action(
    status_row: dict[str, Any] | None,
    graph_path: Path,
    expected_metadata: dict[str, Any],
    retry_failed: bool,
) -> tuple[str, int]:
    """Return run/skip policy without ever accepting a stale completed graph."""
    if status_row is None:
        if graph_path.exists():
            raise RuntimeError(f"Untracked graph already exists: {graph_path}")
        return "run", 1
    status = str(status_row["status"])
    attempt = int(status_row.get("attempt", 1))
    if status == "completed":
        if not graph_path.is_file():
            raise RuntimeError(f"Completed status has no graph: {graph_path}")
        actual = read_graph(graph_path).get("phase4a", {})
        for key in ("manifest_sha256", "encoding_version", "run_id", "parameter_sha256"):
            if actual.get(key) != expected_metadata.get(key):
                raise RuntimeError(f"Completed graph metadata mismatch for {key}: {graph_path}")
        return "skip", attempt
    if status == "excluded_no_valid_building":
        if graph_path.exists():
            raise RuntimeError(f"Excluded case unexpectedly has a graph: {graph_path}")
        return "skip", attempt
    if status == "failed":
        if graph_path.exists():
            raise RuntimeError(f"Failed case unexpectedly has a graph: {graph_path}")
        return ("run", attempt + 1) if retry_failed else ("skip_failed", attempt)
    raise RuntimeError(f"Unsupported prior status: {status!r}")


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _atomic_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    pd.DataFrame(rows, columns=fields).to_csv(
        temporary, index=False, encoding="utf-8-sig", lineterminator="\n"
    )
    os.replace(temporary, path)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_")


def _latest_status(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in rows:
        sample_id = str(row["sample_id"])
        if sample_id not in latest or int(row.get("attempt", 1)) >= int(latest[sample_id].get("attempt", 1)):
            latest[sample_id] = row
    return latest


def _prepare_run(mode: str, run_id: str, chunk_size: int) -> tuple[Path, dict[str, Any]]:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", run_id):
        raise ValueError("run_id contains unsupported characters")
    root = RESULTS_DIR / mode / run_id
    expected = _expected_metadata(run_id, chunk_size)
    preflight_path = root / "preflight.json"
    if root.exists():
        if not preflight_path.is_file():
            raise FileExistsError(f"Run directory exists without resumable preflight: {root}")
        existing = json.loads(preflight_path.read_text(encoding="utf-8"))
        if existing.get("phase4a") != expected:
            raise RuntimeError("Existing run uses a different manifest, version, or parameter snapshot")
    else:
        root.mkdir(parents=True)
        _atomic_json(
            preflight_path,
            {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "mode": mode,
                "run_id": run_id,
                "authorization": "96-case smoke only" if mode == "smoke" else "researcher-started pilot",
                "database_access": "read-only",
                "phase4a": expected,
            },
        )
    return root, expected


def _failure_row(sample: dict, run_id: str, attempt: int, chunk_id: str, error: Exception, query_seconds: float = 0.0) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "attempt": attempt,
        "sample_id": sample["sample_id"],
        "city": sample["city"],
        "block_id": sample["block_id"],
        "city_count_stratum": sample["city_count_stratum"],
        "sample_weight": sample["sample_weight"],
        "chunk_id": chunk_id,
        "status": "failed",
        "planned_building_count": sample["lod1_building_count"],
        "query_seconds": round(query_seconds, 6),
        "runtime_seconds": round(query_seconds, 6),
        "failure_type": type(error).__name__,
        "failure_message": str(error),
    }


def run(mode: str, run_id: str, chunk_size: int, retry_failed: bool) -> dict[str, Any]:
    if mode not in {"smoke", "pilot"}:
        raise ValueError("mode must be smoke or pilot")
    manifest_path = RESULTS_DIR / f"{mode}_manifest.csv"
    rows = _read_rows(manifest_path)
    expected_count = int(SPEC[f"{mode}_rows"])
    if len(rows) != expected_count:
        raise RuntimeError(f"Expected {expected_count} {mode} rows, got {len(rows)}")
    if {row["manifest_sha256"] for row in rows} != {SOURCE_MANIFEST_SHA256}:
        raise RuntimeError("Run manifest does not reference the frozen Phase 3.5B SHA256")

    root, graph_metadata = _prepare_run(mode, run_id, chunk_size)
    status_path = root / "encoding_status.csv"
    chunk_path = root / "chunk_status.csv"
    exclusions_path = root / "building_exclusions.csv"
    status_rows: list[dict[str, Any]] = _read_rows(status_path) if status_path.exists() else []
    chunk_rows: list[dict[str, Any]] = _read_rows(chunk_path) if chunk_path.exists() else []
    exclusion_rows: list[dict[str, Any]] = _read_rows(exclusions_path) if exclusions_path.exists() else []
    latest = _latest_status(status_rows)
    skipped_count = 0
    log_path = root / "run_log.txt"
    run_started = time.perf_counter()
    tracemalloc.start()

    with log_path.open("a", encoding="utf-8") as log, readonly_connection(statement_timeout_ms=600_000) as connection:
        for city, chunk_id, chunk in iter_city_chunks(rows, chunk_size):
            pending: list[tuple[dict, Path, int]] = []
            for sample in chunk:
                graph_path = root / "graphs" / _slug(city) / chunk_id / f"{sample['sample_id']}.graph.json.gz"
                action, attempt = resume_action(latest.get(sample["sample_id"]), graph_path, graph_metadata, retry_failed)
                if action == "run":
                    pending.append((sample, graph_path, attempt))
                else:
                    skipped_count += 1
            if not pending:
                continue
            block_ids = [sample["block_id"] for sample, _, _ in pending]
            query_started = time.perf_counter()
            try:
                blocks_started = time.perf_counter()
                blocks = fetch_block_batch(connection, city, block_ids)
                block_seconds = time.perf_counter() - blocks_started
                missing = sorted(set(block_ids) - set(blocks))
                mappings = {(str(item["city_query"]), int(item["metric_srid"])) for item in blocks.values()}
                if len(mappings) != 1:
                    raise RuntimeError(f"City mapping missing or inconsistent for {city}: {mappings}")
                city_query, metric_srid = next(iter(mappings))
                buildings_started = time.perf_counter()
                buildings = fetch_building_batch(connection, city_query, metric_srid, block_ids)
                building_seconds = time.perf_counter() - buildings_started
                transfer_bytes = estimate_transfer_bytes(blocks, buildings)
                chunk_status = "completed" if not missing else "partial_missing_block"
                chunk_error = "" if not missing else f"Missing blocks: {missing}"
            except Exception as error:
                elapsed = time.perf_counter() - query_started
                for sample, _, attempt in pending:
                    row = _failure_row(sample, run_id, attempt, chunk_id, error, elapsed / len(pending))
                    status_rows.append(row)
                    latest[sample["sample_id"]] = row
                    _atomic_csv(status_path, status_rows, STATUS_FIELDS)
                chunk_rows.append(
                    {
                        "run_id": run_id, "city": city, "chunk_id": chunk_id,
                        "planned_blocks": len(pending), "returned_blocks": 0,
                        "returned_buildings": 0, "query_seconds": round(elapsed, 6),
                        "status": "failed", "failure_message": str(error),
                    }
                )
                _atomic_csv(chunk_path, chunk_rows, CHUNK_FIELDS)
                log.write(f"{city}/{chunk_id} query FAILED {error}\n")
                log.flush()
                continue

            chunk_rows.append(
                {
                    "run_id": run_id, "city": city, "chunk_id": chunk_id,
                    "planned_blocks": len(pending), "returned_blocks": len(blocks),
                    "returned_buildings": sum(len(value) for value in buildings.values()),
                    "block_query_seconds": round(block_seconds, 6),
                    "building_query_seconds": round(building_seconds, 6),
                    "query_seconds": round(block_seconds + building_seconds, 6),
                    "estimated_transfer_bytes": transfer_bytes,
                    "status": chunk_status, "failure_message": chunk_error,
                }
            )
            _atomic_csv(chunk_path, chunk_rows, CHUNK_FIELDS)
            count_sum = sum(max(1, int(sample["lod1_building_count"])) for sample, _, _ in pending)
            for sample, graph_path, attempt in pending:
                planned_count = int(sample["lod1_building_count"])
                allocated_query = block_seconds / len(pending) + building_seconds * max(1, planned_count) / count_sum
                if sample["block_id"] not in blocks:
                    row = _failure_row(
                        sample, run_id, attempt, chunk_id,
                        RuntimeError("Manifest block missing from database"), allocated_query,
                    )
                    status_rows.append(row)
                    latest[sample["sample_id"]] = row
                    _atomic_csv(status_path, status_rows, STATUS_FIELDS)
                    continue
                started = time.perf_counter()
                tracemalloc.reset_peak()
                try:
                    block = blocks[sample["block_id"]]
                    case_buildings = buildings.get(sample["block_id"], [])
                    outcome = encode_block_to_v031(
                        sample,
                        block,
                        case_buildings,
                        sample_manifest_sha256=SOURCE_MANIFEST_SHA256,
                        tessellation_config=DEFAULT_TESSELLATION_CONFIG,
                    )
                    encoding_seconds = time.perf_counter() - started
                    _, peak = tracemalloc.get_traced_memory()
                    graph_path_value = ""
                    output_bytes = 0
                    node_count = edge_count = cell_count = 0
                    if outcome.status == "completed":
                        if outcome.record is None or not outcome.record["validation"]["passed"]:
                            raise RuntimeError("v0.3.1 encoder returned an invalid completed graph")
                        outcome.record["phase4a"] = graph_metadata
                        atomic_write_graph(outcome.record, graph_path)
                        graph_path_value = graph_path.relative_to(PHASE_DIR).as_posix()
                        output_bytes = graph_path.stat().st_size
                        node_count = len(outcome.record["nodes"])
                        edge_count = len(outcome.record["edges"])
                        cell_count = len(outcome.record["cells"])
                    for item in outcome.excluded_buildings:
                        exclusion_rows.append({"run_id": run_id, "attempt": attempt, **item})
                    row = {
                        "run_id": run_id, "attempt": attempt, "sample_id": sample["sample_id"],
                        "city": city, "block_id": sample["block_id"],
                        "city_count_stratum": sample["city_count_stratum"],
                        "sample_weight": sample["sample_weight"], "chunk_id": chunk_id,
                        "status": outcome.status, "planned_building_count": planned_count,
                        "actual_building_count": len(case_buildings),
                        "building_count_matches": len(case_buildings) == planned_count,
                        "query_seconds": round(allocated_query, 6),
                        "encoding_seconds": round(encoding_seconds, 6),
                        "runtime_seconds": round(allocated_query + encoding_seconds, 6),
                        "node_count": node_count, "edge_count": edge_count, "cell_count": cell_count,
                        "excluded_building_count": len(outcome.excluded_buildings),
                        "output_bytes": output_bytes,
                        "python_peak_memory_mb": round(peak / 1024**2, 3),
                        "graph_path": graph_path_value, "failure_type": "", "failure_message": "",
                    }
                except Exception as error:
                    encoding_seconds = time.perf_counter() - started
                    _, peak = tracemalloc.get_traced_memory()
                    row = _failure_row(sample, run_id, attempt, chunk_id, error, allocated_query)
                    row.update(
                        {
                            "encoding_seconds": round(encoding_seconds, 6),
                            "runtime_seconds": round(allocated_query + encoding_seconds, 6),
                            "python_peak_memory_mb": round(peak / 1024**2, 3),
                        }
                    )
                status_rows.append(row)
                latest[sample["sample_id"]] = row
                _atomic_csv(status_path, status_rows, STATUS_FIELDS)
                _atomic_csv(exclusions_path, exclusion_rows, EXCLUSION_FIELDS)
                message = f"{city}/{chunk_id} {sample['sample_id']} {row['status']}"
                print(message, flush=True)
                log.write(message + "\n")
                log.flush()
    tracemalloc.stop()

    latest_rows = list(_latest_status(status_rows).values())
    counts = pd.Series([row["status"] for row in latest_rows]).value_counts().to_dict()
    summary = {
        "run_id": run_id,
        "mode": mode,
        "planned": len(rows),
        "completed": int(counts.get("completed", 0)),
        "excluded_no_valid_building": int(counts.get("excluded_no_valid_building", 0)),
        "failed": int(counts.get("failed", 0)),
        "interrupted": int(len(rows) - len(latest_rows)),
        "skipped": skipped_count,
        "total_wall_seconds_this_invocation": round(time.perf_counter() - run_started, 6),
        "database_access": "read-only",
        "full_run_started": False,
    }
    _atomic_json(root / "run_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "pilot"), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()
    result = run(args.mode, args.run_id, args.chunk_size, args.retry_failed)
    return 0 if result["interrupted"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
