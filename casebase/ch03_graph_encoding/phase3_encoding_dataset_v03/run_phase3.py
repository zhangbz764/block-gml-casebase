"""Pilot and full batch runner for the frozen phase 3 manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
import tracemalloc
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from psycopg2 import sql
from shapely import from_wkb


PHASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PHASE_DIR / "results"
CH03_DIR = PHASE_DIR.parent
PHASE2_DIR = CH03_DIR / "phase2_encoding_v01"
sys.path.insert(0, str(CH03_DIR))
sys.path.insert(0, str(PHASE2_DIR))

from shared.db_readonly import readonly_connection
from export_phase3 import (
    DEFAULT_TESSELLATION_CONFIG,
    encode_block_to_v03,
    read_graph,
    validate_v03_record,
    write_graph,
)
from run_phase2 import METRIC_FIELDS


def select_pilot_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Choose one frozen core sample per city/stratum and every regression anchor."""
    ordered = sorted(rows, key=lambda row: (row["city"], row["block_id"]))
    selected: dict[tuple[str, str], dict[str, str]] = {}
    core_keys: set[tuple[str, int]] = set()
    for row in ordered:
        groups = row["sample_groups"].split(";")
        if "core_audit" not in groups:
            continue
        key = row["city"], int(row["selection_stratum"])
        if key not in core_keys:
            core_keys.add(key)
            selected[(row["city"], row["block_id"])] = row
    for row in ordered:
        if "regression_anchor" in row["sample_groups"].split(";"):
            selected[(row["city"], row["block_id"])] = row
    return sorted(
        selected.values(),
        key=lambda row: (
            0 if "core_audit" in row["sample_groups"].split(";") else 1,
            row["city"],
            row["block_id"],
        ),
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_blocks(connection, rows: list[dict[str, str]]) -> dict[str, dict[str, Any]]:
    block_ids = [row["block_id"] for row in rows]
    selected_columns = sql.SQL(", ").join(
        sql.SQL("v.{} AS {}").format(sql.Identifier(name), sql.Identifier(name))
        for name in METRIC_FIELDS
    )
    statement = sql.SQL(
        """
        SELECT v.block_id, v.city, v.country, m.city_query,
               m.lod1_metric_srid AS metric_srid, ST_SRID(v.geom) AS storage_srid,
               {selected_columns},
               ST_AsBinary(ST_Transform(ST_Force2D(v.geom), m.lod1_metric_srid)) AS geometry_wkb_metric
        FROM summary.lod1_candidate_blocks v
        JOIN summary.city_name_map m ON m.city = v.city
        WHERE v.block_id = ANY(%s)
        ORDER BY v.block_id
        """
    ).format(selected_columns=selected_columns)
    with connection.cursor() as cursor:
        cursor.execute(statement, (block_ids,))
        columns = [item.name for item in cursor.description]
        fetched = [dict(zip(columns, item)) for item in cursor.fetchall()]
    blocks = {}
    for block in fetched:
        block["geometry"] = from_wkb(bytes(block.pop("geometry_wkb_metric")))
        blocks[str(block["block_id"])] = block
    missing = sorted(set(block_ids) - set(blocks))
    if missing:
        raise RuntimeError(f"Manifest blocks missing from database: {missing}")
    return blocks


def fetch_buildings(connection, block: dict[str, Any]) -> list[dict[str, Any]]:
    relation = f"{block['city_query']}_buildings_lod1"
    statement = sql.SQL(
        """
        SELECT building_id, citygml_id, height, area, perimeter,
               ST_AsBinary(ST_Transform(ST_Force2D(geom_2d), %s)) AS geometry_wkb_metric
        FROM lod1.{}
        WHERE block_id = %s
        ORDER BY building_id
        """
    ).format(sql.Identifier(relation))
    with connection.cursor() as cursor:
        cursor.execute(statement, (block["metric_srid"], block["block_id"]))
        columns = [item.name for item in cursor.description]
        buildings = [dict(zip(columns, item)) for item in cursor.fetchall()]
    for building in buildings:
        building["geometry"] = from_wkb(bytes(building.pop("geometry_wkb_metric")))
    return buildings


def _structure_signature(record: dict[str, Any]) -> dict[str, list[str]]:
    return {
        "nodes": [str(item["building_id"]) for item in record["nodes"]],
        "cells": [str(item["building_id"]) for item in record["cells"]],
        "edges": [str(item["edge_id"]) for item in record["edges"]],
    }


def run_batch(mode: str) -> dict[str, Any]:
    manifest_path = RESULTS_DIR / "sample_manifest.csv"
    all_rows = _read_csv(manifest_path)
    rows = select_pilot_rows(all_rows) if mode == "pilot" else all_rows
    snapshot = json.loads((RESULTS_DIR / "database_snapshot.json").read_text(encoding="utf-8"))
    manifest_hash = snapshot["sample_manifest_sha256"]
    if _sha256(manifest_path) != manifest_hash:
        raise RuntimeError("sample_manifest.csv hash no longer matches database_snapshot.json")

    output_root = RESULTS_DIR / ("pilot" if mode == "pilot" else "dataset")
    graph_dir = output_root / "graphs"
    status_path = (
        output_root / "encoding_status.csv"
        if mode == "pilot"
        else RESULTS_DIR / "audit" / "encoding_status.csv"
    )
    if status_path.exists():
        raise RuntimeError(f"Refusing to overwrite prior batch status: {status_path}")
    graph_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"{mode}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    log_path = output_root / "run_log.txt"
    status_rows: list[dict[str, Any]] = []
    run_started = time.perf_counter()
    tracemalloc.start()

    with log_path.open("w", encoding="utf-8") as log, readonly_connection(statement_timeout_ms=600_000) as connection:
        blocks = fetch_blocks(connection, rows)
        for index, sample in enumerate(rows, start=1):
            started = time.perf_counter()
            tracemalloc.reset_peak()
            graph_path = graph_dir / f"{sample['sample_id']}.graph.json.gz"
            base = {
                "run_id": run_id,
                "attempt": 1,
                "sample_id": sample["sample_id"],
                "city": sample["city"],
                "block_id": sample["block_id"],
                "sample_groups": sample["sample_groups"],
                "planned_building_count": sample["lod1_building_count"],
            }
            try:
                block = blocks[sample["block_id"]]
                buildings = fetch_buildings(connection, block)
                record = encode_block_to_v03(
                    sample,
                    block,
                    buildings,
                    sample_manifest_sha256=manifest_hash,
                    tessellation_config=DEFAULT_TESSELLATION_CONFIG,
                )
                validation = validate_v03_record(record)
                if not validation["passed"]:
                    raise ValueError(json.dumps(validation["issues"], ensure_ascii=False))
                write_graph(record, graph_path)
                reloaded = read_graph(graph_path)
                roundtrip_ok = _structure_signature(record) == _structure_signature(reloaded)
                if not roundtrip_ok:
                    raise ValueError("graph structure signature changed after write/read")
                _, peak = tracemalloc.get_traced_memory()
                elapsed = time.perf_counter() - started
                status_rows.append(
                    {
                        **base,
                        "status": "completed",
                        "actual_building_count": len(buildings),
                        "node_count": len(record["nodes"]),
                        "cell_count": len(record["cells"]),
                        "cell_failure_count": len(record["cell_failures"]),
                        "edge_count": len(record["edges"]),
                        "sample_point_count": record["diagnostics"]["tessellation"]["sample_point_count"],
                        "long_distance_candidate_count": record["weak_edge_summary"]["long_distance_candidate_count"],
                        "low_shared_candidate_count": record["weak_edge_summary"]["low_shared_candidate_count"],
                        "combined_weak_candidate_count": record["weak_edge_summary"]["combined_weak_candidate_count"],
                        "runtime_seconds": round(elapsed, 6),
                        "output_bytes": graph_path.stat().st_size,
                        "python_peak_memory_mb": round(peak / (1024 * 1024), 3),
                        "roundtrip_ok": True,
                        "quality_flags": ";".join(record["graph"]["quality_flags"]),
                        "failure_message": "",
                    }
                )
                message = f"[{index}/{len(rows)}] {sample['sample_id']} completed nodes={len(buildings)} edges={len(record['edges'])} seconds={elapsed:.3f}"
            except Exception as error:
                _, peak = tracemalloc.get_traced_memory()
                elapsed = time.perf_counter() - started
                status_rows.append(
                    {
                        **base,
                        "status": "failed",
                        "runtime_seconds": round(elapsed, 6),
                        "python_peak_memory_mb": round(peak / (1024 * 1024), 3),
                        "roundtrip_ok": False,
                        "failure_message": str(error),
                    }
                )
                message = f"[{index}/{len(rows)}] {sample['sample_id']} FAILED {error}"
            print(message, flush=True)
            log.write(message + "\n")
            log.flush()
    tracemalloc.stop()

    fields = [
        "run_id", "attempt", "sample_id", "city", "block_id", "sample_groups",
        "status", "planned_building_count", "actual_building_count", "node_count",
        "cell_count", "cell_failure_count", "edge_count", "sample_point_count",
        "long_distance_candidate_count", "low_shared_candidate_count",
        "combined_weak_candidate_count", "runtime_seconds", "output_bytes",
        "python_peak_memory_mb", "roundtrip_ok", "quality_flags", "failure_message",
    ]
    _write_csv(status_path, status_rows, fields)
    total_seconds = time.perf_counter() - run_started
    completed = [row for row in status_rows if row["status"] == "completed"]
    total_bytes = sum(int(row.get("output_bytes") or 0) for row in completed)
    projected_seconds = total_seconds / len(rows) * len(all_rows)
    projected_bytes = total_bytes / max(1, len(completed)) * len(all_rows)
    summary = {
        "run_id": run_id,
        "mode": mode,
        "planned_samples": len(rows),
        "completed_samples": len(completed),
        "failed_samples": len(rows) - len(completed),
        "total_runtime_seconds": total_seconds,
        "total_graph_bytes": total_bytes,
        "projected_full_runtime_seconds": projected_seconds,
        "projected_full_graph_bytes": projected_bytes,
        "resource_gate_runtime_limit_seconds": 8 * 3600,
        "resource_gate_storage_limit_bytes": 20 * 1024**3,
        "resource_gate_passed": projected_seconds <= 8 * 3600 and projected_bytes <= 20 * 1024**3,
        "sample_manifest_sha256": manifest_hash,
    }
    (output_root / f"{mode}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    freeze = {
        "run_id": run_id,
        "sample_manifest_sha256": manifest_hash,
        "export_phase3_sha256": _sha256(PHASE_DIR / "export_phase3.py"),
        "run_phase3_sha256": _sha256(PHASE_DIR / "run_phase3.py"),
        "tessellation_parameters": DEFAULT_TESSELLATION_CONFIG.__dict__,
    }
    (output_root / "freeze.json").write_text(
        json.dumps(freeze, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if mode == "pilot":
        _write_csv(output_root / "pilot_manifest.csv", rows, list(rows[0]))
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("pilot", "full"), default="pilot")
    args = parser.parse_args()
    summary = run_batch(args.mode)
    return 0 if summary["failed_samples"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
