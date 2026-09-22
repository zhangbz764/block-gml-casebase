"""Run exactly one frozen Phase 4B part with read-only, recoverable batching."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd


PHASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PHASE_DIR / "results"
CH03_DIR = PHASE_DIR.parent
PHASE4A_DIR = CH03_DIR / "phase4a_chunked_graph_pilot_v01"
PHASE3_DIR = CH03_DIR / "phase3_encoding_dataset_v03"
sys.path.insert(0, str(CH03_DIR))
sys.path.insert(0, str(PHASE4A_DIR))
sys.path.insert(0, str(PHASE3_DIR))

from batch_db_reader import estimate_transfer_bytes, fetch_block_batch, fetch_building_batch, iter_city_chunks
from export_phase3 import DEFAULT_TESSELLATION_CONFIG, read_graph, write_graph
from shared.db_readonly import readonly_connection

from analyze_phase4b import analyze_part
from export_phase4b_v032 import ENCODING_VERSION, encode_block_to_v032


SPEC_PATH = PHASE_DIR / "encoding_spec_v032.json"
SPEC = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
PART_IDS = tuple(f"part_{index:02d}" for index in range(1, 13))
STATUS_FIELDS = [
    "run_id", "part_id", "attempt", "sample_id", "city", "block_id", "chunk_id",
    "status", "planned_building_count", "actual_building_count", "building_count_matches",
    "query_seconds", "encoding_seconds", "runtime_seconds", "node_count", "cell_count",
    "edge_count", "excluded_building_count", "output_bytes", "graph_path",
    "failure_type", "failure_message",
]
CHUNK_FIELDS = [
    "run_id", "part_id", "city", "chunk_id", "planned_blocks", "returned_blocks",
    "returned_buildings", "block_query_seconds", "building_query_seconds", "query_seconds",
    "estimated_transfer_bytes", "status", "failure_message",
]
EXCLUSION_FIELDS = [
    "run_id", "part_id", "attempt", "sample_id", "city", "block_id", "building_id",
    "reason", "footprint_area_m2", "footprint_perimeter_m", "footprint_compactness",
    "geometry_was_valid", "geometry_validity_detail", "footprint_covered_by_block",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    pd.DataFrame(rows, columns=fields).to_csv(temporary, index=False, encoding="utf-8-sig", lineterminator="\n")
    os.replace(temporary, path)


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    return pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False).to_dict("records")


def _latest(rows: list[dict]) -> dict[str, dict]:
    latest: dict[str, dict] = {}
    for row in rows:
        key = str(row["sample_id"])
        if key not in latest or int(row.get("attempt", 1)) >= int(latest[key].get("attempt", 1)):
            latest[key] = row
    return latest


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_")


def validate_part_rows(rows: list[dict], part_id: str) -> None:
    actual = {str(row.get("part_id")) for row in rows}
    if actual != {part_id}:
        raise RuntimeError(f"Part manifest crossed part boundary: expected {part_id}, got {sorted(actual)}")
    if len({(row["city"], row["block_id"]) for row in rows}) != len(rows):
        raise RuntimeError("Part manifest contains duplicate city + block_id")


def _metadata(part_id: str, run_id: str, chunk_size: int, part_path: Path) -> dict:
    hashes = json.loads((RESULTS_DIR / "audit" / "prep" / "manifest_hashes.json").read_text(encoding="utf-8"))
    parameters = {"chunk_size": chunk_size, "tessellation": dict(DEFAULT_TESSELLATION_CONFIG.__dict__)}
    return {
        "source_manifest_sha256": SPEC["source_manifest_sha256"],
        "full_manifest_sha256": hashes["full_run_manifest_v032.csv.gz"],
        "part_manifest_sha256": sha256_file(part_path),
        "encoding_spec_sha256": sha256_file(SPEC_PATH),
        "encoding_version": ENCODING_VERSION,
        "part_id": part_id,
        "run_id": run_id,
        "parameter_sha256": hashlib.sha256(json.dumps(parameters, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "parameters": parameters,
    }


def atomic_write_graph(record: dict[str, Any], path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite graph: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.gz")
    if temporary.exists():
        temporary.unlink()
    try:
        write_graph(record, temporary)
        reloaded = read_graph(temporary)
        if reloaded.get("phase4b") != record.get("phase4b") or not reloaded.get("validation", {}).get("passed"):
            raise RuntimeError("Graph failed metadata or validation roundtrip")
        for field in ("nodes", "cells", "edges"):
            if len(reloaded.get(field, [])) != len(record.get(field, [])):
                raise RuntimeError(f"Graph {field} count changed during roundtrip")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def resume_action(status_row: dict | None, graph_path: Path, expected: dict, retry_failed: bool) -> tuple[str, int]:
    if status_row is None:
        if graph_path.exists():
            raise RuntimeError(f"Untracked graph exists: {graph_path}")
        return "run", 1
    status = str(status_row["status"])
    attempt = int(status_row.get("attempt", 1))
    if status == "completed":
        if not graph_path.is_file():
            raise RuntimeError(f"Completed graph is missing: {graph_path}")
        actual = read_graph(graph_path).get("phase4b", {})
        for key in ("source_manifest_sha256", "full_manifest_sha256", "part_manifest_sha256", "encoding_spec_sha256", "encoding_version", "part_id", "run_id", "parameter_sha256"):
            if actual.get(key) != expected.get(key):
                raise RuntimeError(f"Completed graph metadata mismatch for {key}")
        return "skip", attempt
    if status == "excluded_no_valid_building":
        if graph_path.exists():
            raise RuntimeError("Excluded case unexpectedly has a graph")
        return "skip", attempt
    if status == "failed":
        if graph_path.exists():
            raise RuntimeError("Failed case unexpectedly has a graph")
        return ("run", attempt + 1) if retry_failed else ("skip_failed", attempt)
    raise RuntimeError(f"Unsupported status: {status}")


def _failure(sample: dict, run_id: str, part_id: str, attempt: int, chunk_id: str, error: Exception, query: float = 0.0) -> dict:
    return {
        "run_id": run_id, "part_id": part_id, "attempt": attempt,
        "sample_id": sample["sample_id"], "city": sample["city"], "block_id": sample["block_id"],
        "chunk_id": chunk_id, "status": "failed", "planned_building_count": sample["lod1_building_count"],
        "query_seconds": round(query, 6), "runtime_seconds": round(query, 6),
        "failure_type": type(error).__name__, "failure_message": str(error),
    }


def run_part(part_id: str, run_id: str, chunk_size: int, retry_failed: bool) -> dict:
    if chunk_size != 100:
        raise ValueError("Phase 4B chunk_size is frozen at 100")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", run_id):
        raise ValueError("run_id contains unsupported characters")
    part_path = RESULTS_DIR / "dataset" / "parts" / f"{part_id}_manifest.csv.gz"
    rows = _read_csv_rows(part_path)
    validate_part_rows(rows, part_id)
    metadata = _metadata(part_id, run_id, chunk_size, part_path)
    run_dir = RESULTS_DIR / "runs" / "full_v032" / part_id
    preflight = run_dir / "preflight.json"
    if run_dir.exists():
        if not preflight.is_file() or json.loads(preflight.read_text(encoding="utf-8")).get("phase4b") != metadata:
            raise RuntimeError("Existing part run has incompatible or missing preflight metadata")
    else:
        run_dir.mkdir(parents=True)
        _atomic_json(preflight, {"phase4b": metadata, "database_access": "read-only", "formal_part_run": True})
    status_path = run_dir / "encoding_status.csv"
    chunk_path = run_dir / "chunk_status.csv"
    exclusion_path = run_dir / "building_exclusions.csv"
    statuses = _read_csv_rows(status_path) if status_path.exists() else []
    chunks = _read_csv_rows(chunk_path) if chunk_path.exists() else []
    exclusions = _read_csv_rows(exclusion_path) if exclusion_path.exists() else []
    latest = _latest(statuses)
    skipped = 0
    log = (run_dir / "run_log.txt").open("a", encoding="utf-8")
    try:
        with readonly_connection(statement_timeout_ms=600_000) as connection:
            for city, chunk_id, chunk in iter_city_chunks(rows, chunk_size):
                pending = []
                for sample in chunk:
                    graph_path = RESULTS_DIR / "dataset" / "graphs" / part_id / _slug(city) / chunk_id / f"{sample['sample_id']}.graph.json.gz"
                    action, attempt = resume_action(latest.get(sample["sample_id"]), graph_path, metadata, retry_failed)
                    if action == "run":
                        pending.append((sample, graph_path, attempt))
                    else:
                        skipped += 1
                if not pending:
                    continue
                ids = [sample["block_id"] for sample, _, _ in pending]
                query_started = time.perf_counter()
                try:
                    started = time.perf_counter(); blocks = fetch_block_batch(connection, city, ids); block_s = time.perf_counter() - started
                    mapping = {(str(row["city_query"]), int(row["metric_srid"])) for row in blocks.values()}
                    if len(mapping) != 1:
                        raise RuntimeError(f"Missing or inconsistent city mapping for {city}")
                    city_query, srid = next(iter(mapping))
                    started = time.perf_counter(); buildings = fetch_building_batch(connection, city_query, srid, ids); building_s = time.perf_counter() - started
                    transfer = estimate_transfer_bytes(blocks, buildings)
                except Exception as error:
                    elapsed = time.perf_counter() - query_started
                    for sample, _, attempt in pending:
                        row = _failure(sample, run_id, part_id, attempt, chunk_id, error, elapsed / len(pending))
                        statuses.append(row); latest[sample["sample_id"]] = row; _atomic_csv(status_path, statuses, STATUS_FIELDS)
                    chunks.append({"run_id": run_id, "part_id": part_id, "city": city, "chunk_id": chunk_id, "planned_blocks": len(pending), "returned_blocks": 0, "returned_buildings": 0, "query_seconds": round(elapsed, 6), "status": "failed", "failure_message": str(error)})
                    _atomic_csv(chunk_path, chunks, CHUNK_FIELDS)
                    continue
                chunks.append({"run_id": run_id, "part_id": part_id, "city": city, "chunk_id": chunk_id, "planned_blocks": len(pending), "returned_blocks": len(blocks), "returned_buildings": sum(map(len, buildings.values())), "block_query_seconds": round(block_s, 6), "building_query_seconds": round(building_s, 6), "query_seconds": round(block_s + building_s, 6), "estimated_transfer_bytes": transfer, "status": "completed", "failure_message": ""})
                _atomic_csv(chunk_path, chunks, CHUNK_FIELDS)
                weight_total = sum(max(1, int(sample["lod1_building_count"])) for sample, _, _ in pending)
                for sample, graph_path, attempt in pending:
                    planned = int(sample["lod1_building_count"])
                    query_s = block_s / len(pending) + building_s * max(1, planned) / weight_total
                    started = time.perf_counter()
                    try:
                        if sample["block_id"] not in blocks:
                            raise RuntimeError("Manifest block missing from database")
                        case_buildings = buildings.get(sample["block_id"], [])
                        outcome = encode_block_to_v032(sample, blocks[sample["block_id"]], case_buildings, sample_manifest_sha256=metadata["full_manifest_sha256"], tessellation_config=DEFAULT_TESSELLATION_CONFIG)
                        encoding_s = time.perf_counter() - started
                        graph_rel = ""; output_bytes = 0; nodes = cells = edges = 0
                        if outcome.status == "completed":
                            if outcome.record is None or not outcome.record["validation"]["passed"]:
                                raise RuntimeError("v0.3.2 returned invalid completed graph")
                            outcome.record["phase4b"] = metadata
                            atomic_write_graph(outcome.record, graph_path)
                            graph_rel = graph_path.relative_to(PHASE_DIR).as_posix(); output_bytes = graph_path.stat().st_size
                            nodes = len(outcome.record["nodes"]); cells = len(outcome.record["cells"]); edges = len(outcome.record["edges"])
                        for item in outcome.excluded_buildings:
                            exclusions.append({"run_id": run_id, "part_id": part_id, "attempt": attempt, **item})
                        row = {"run_id": run_id, "part_id": part_id, "attempt": attempt, "sample_id": sample["sample_id"], "city": city, "block_id": sample["block_id"], "chunk_id": chunk_id, "status": outcome.status, "planned_building_count": planned, "actual_building_count": len(case_buildings), "building_count_matches": len(case_buildings) == planned, "query_seconds": round(query_s, 6), "encoding_seconds": round(encoding_s, 6), "runtime_seconds": round(query_s + encoding_s, 6), "node_count": nodes, "cell_count": cells, "edge_count": edges, "excluded_building_count": len(outcome.excluded_buildings), "output_bytes": output_bytes, "graph_path": graph_rel, "failure_type": "", "failure_message": ""}
                    except Exception as error:
                        encoding_s = time.perf_counter() - started
                        row = _failure(sample, run_id, part_id, attempt, chunk_id, error, query_s)
                        row.update({"encoding_seconds": round(encoding_s, 6), "runtime_seconds": round(query_s + encoding_s, 6)})
                    statuses.append(row); latest[sample["sample_id"]] = row
                    _atomic_csv(status_path, statuses, STATUS_FIELDS); _atomic_csv(exclusion_path, exclusions, EXCLUSION_FIELDS)
                    message = f"{part_id} {city}/{chunk_id} {sample['sample_id']} {row['status']}"; print(message, flush=True); log.write(message + "\n"); log.flush()
    finally:
        log.close()
    summary = analyze_part(part_id, len(rows), skipped=skipped)
    _atomic_json(run_dir / "run_summary.json", {"run_id": run_id, "part_id": part_id, **summary})
    print(json.dumps(summary, ensure_ascii=False))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part-id", required=True, choices=PART_IDS)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--retry-failed", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = run_part(args.part_id, args.run_id, args.chunk_size, args.retry_failed)
    return 0 if result["interrupted"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
