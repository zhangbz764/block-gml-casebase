"""Rebuild only the 57 Phase 3 graphs affected by collapsed footprints."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PHASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PHASE_DIR / "results"
CH03_DIR = PHASE_DIR.parent
PHASE3_DIR = CH03_DIR / "phase3_encoding_dataset_v03"
sys.path.insert(0, str(PHASE_DIR))
sys.path.insert(0, str(PHASE3_DIR))

from export_phase3 import DEFAULT_TESSELLATION_CONFIG, read_graph, write_graph
from export_phase3_1 import encode_block_to_v031
from run_phase3 import (
    _read_csv,
    _sha256,
    _structure_signature,
    _write_csv,
    fetch_blocks,
    fetch_buildings,
)

sys.path.insert(0, str(CH03_DIR))
from shared.db_readonly import readonly_connection


BASELINE_RESULTS = PHASE3_DIR / "results"
AFFECTED_MANIFEST = RESULTS_DIR / "affected_manifest.csv"
REGRESSION_MANIFEST = RESULTS_DIR / "regression_manifest.csv"
GRAPH_DIR = RESULTS_DIR / "dataset" / "graphs"
AUDIT_DIR = RESULTS_DIR / "audit"


SUMMARY_FIELDS = [
    "run_id", "sample_id", "city", "block_id", "status",
    "source_building_count", "encoded_building_count", "excluded_building_count",
    "excluded_footprint_area_m2", "excluded_footprint_area_ratio",
    "baseline_node_count", "baseline_cell_count", "baseline_cell_failure_count",
    "node_count", "cell_count", "cell_failure_count", "edge_count",
    "roundtrip_ok", "validation_passed", "graph_path", "runtime_seconds",
    "failure_message",
]

EXCLUSION_FIELDS = [
    "sample_id", "city", "block_id", "building_id", "reason",
    "footprint_area_m2", "footprint_perimeter_m", "footprint_compactness",
    "footprint_covered_by_block",
]

REGRESSION_FIELDS = [
    "run_id", "sample_id", "city", "block_id", "status",
    "excluded_building_count", "node_signature_match", "cell_signature_match",
    "edge_signature_match", "all_match", "baseline_node_count",
    "baseline_cell_count", "baseline_edge_count", "new_node_count",
    "new_cell_count", "new_edge_count", "failure_message",
]


def _relative_to_ch03(path: Path) -> str:
    return path.relative_to(CH03_DIR).as_posix()


def _all_excluded_summary(buildings: list[dict[str, Any]], exclusions: list[dict[str, Any]]) -> dict[str, Any]:
    excluded_area = sum(float(item["footprint_area_m2"]) for item in exclusions)
    return {
        "source_building_count": len(buildings),
        "encoded_building_count": 0,
        "excluded_building_count": len(exclusions),
        "excluded_footprint_area_m2": excluded_area,
        "excluded_footprint_area_ratio": 1.0 if excluded_area > 0 else None,
    }


def _library_versions() -> dict[str, str]:
    names = ("shapely", "scipy", "pandas", "geopandas", "matplotlib", "numpy")
    return {name: importlib.metadata.version(name) for name in names}


def run_affected() -> dict[str, Any]:
    affected = _read_csv(AFFECTED_MANIFEST)
    regression = _read_csv(REGRESSION_MANIFEST)
    if len(affected) != 57 or len(regression) != 3:
        raise RuntimeError("Frozen Phase 3.1 manifest sizes changed")

    snapshot = json.loads((BASELINE_RESULTS / "database_snapshot.json").read_text(encoding="utf-8"))
    manifest_hash = snapshot["sample_manifest_sha256"]
    phase3_manifest = BASELINE_RESULTS / "sample_manifest.csv"
    if _sha256(phase3_manifest) != manifest_hash:
        raise RuntimeError("Frozen Phase 3 manifest hash changed")

    summary_path = AUDIT_DIR / "revised_graph_summary.csv"
    regression_path = AUDIT_DIR / "regression_signature_check.csv"
    exclusions_path = AUDIT_DIR / "excluded_buildings.csv"
    occupied = [path for path in (summary_path, regression_path, exclusions_path) if path.exists()]
    existing_graphs = list(GRAPH_DIR.glob("*.graph.json.gz")) if GRAPH_DIR.exists() else []
    if occupied or existing_graphs:
        raise RuntimeError(f"Refusing to overwrite Phase 3.1 outputs: {occupied + existing_graphs[:3]}")

    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    run_id = f"affected-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    summary_rows: list[dict[str, Any]] = []
    exclusion_rows: list[dict[str, Any]] = []
    regression_rows: list[dict[str, Any]] = []
    log_path = RESULTS_DIR / "run_log.txt"

    with log_path.open("w", encoding="utf-8") as log, readonly_connection(statement_timeout_ms=600_000) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT current_database(), version()")
            database_name, database_version = cursor.fetchone()
        all_rows = affected + regression
        blocks = fetch_blocks(connection, all_rows)
        log.write(f"run_id={run_id}\n")
        log.write(f"phase3_manifest={phase3_manifest}\n")
        log.write(f"sample_manifest_sha256={manifest_hash}\n")
        log.write(f"baseline_graph_dir={BASELINE_RESULTS / 'dataset' / 'graphs'}\n")

        for index, sample in enumerate(affected, start=1):
            started = time.perf_counter()
            graph_path = GRAPH_DIR / f"{sample['sample_id']}.graph.json.gz"
            base = {
                "run_id": run_id,
                "sample_id": sample["sample_id"],
                "city": sample["city"],
                "block_id": sample["block_id"],
                "baseline_node_count": sample["baseline_node_count"],
                "baseline_cell_count": sample["baseline_cell_count"],
                "baseline_cell_failure_count": sample["baseline_cell_failure_count"],
            }
            try:
                block = blocks[sample["block_id"]]
                buildings = fetch_buildings(connection, block)
                if len(buildings) != int(sample["lod1_building_count"]):
                    raise RuntimeError(
                        f"Frozen building count changed: expected {sample['lod1_building_count']}, got {len(buildings)}"
                    )
                outcome = encode_block_to_v031(
                    sample,
                    block,
                    buildings,
                    sample_manifest_sha256=manifest_hash,
                    tessellation_config=DEFAULT_TESSELLATION_CONFIG,
                )
                exclusion_rows.extend(outcome.excluded_buildings)
                if outcome.status == "excluded_no_valid_building":
                    if graph_path.exists():
                        raise RuntimeError("All-excluded case unexpectedly has a graph file")
                    graph_summary = _all_excluded_summary(buildings, outcome.excluded_buildings)
                    summary_rows.append(
                        {
                            **base, **graph_summary, "status": outcome.status,
                            "node_count": 0, "cell_count": 0, "cell_failure_count": 0,
                            "edge_count": 0, "roundtrip_ok": "", "validation_passed": "",
                            "graph_path": "", "runtime_seconds": round(time.perf_counter() - started, 6),
                            "failure_message": "",
                        }
                    )
                else:
                    record = outcome.record
                    if record is None:
                        raise RuntimeError("Completed outcome has no graph record")
                    if not record["validation"]["passed"]:
                        raise RuntimeError(json.dumps(record["validation"]["issues"], ensure_ascii=False))
                    write_graph(record, graph_path)
                    reloaded = read_graph(graph_path)
                    roundtrip_ok = _structure_signature(record) == _structure_signature(reloaded)
                    if not roundtrip_ok:
                        raise RuntimeError("Graph structure signature changed after compression roundtrip")
                    graph = record["graph"]
                    summary_rows.append(
                        {
                            **base,
                            "status": outcome.status,
                            "source_building_count": graph["source_building_count"],
                            "encoded_building_count": graph["encoded_building_count"],
                            "excluded_building_count": graph["excluded_building_count"],
                            "excluded_footprint_area_m2": graph["excluded_footprint_area_m2"],
                            "excluded_footprint_area_ratio": graph["excluded_footprint_area_ratio"],
                            "node_count": graph["node_count"], "cell_count": graph["cell_count"],
                            "cell_failure_count": graph["cell_failure_count"], "edge_count": graph["edge_count"],
                            "roundtrip_ok": True, "validation_passed": True,
                            "graph_path": _relative_to_ch03(graph_path),
                            "runtime_seconds": round(time.perf_counter() - started, 6),
                            "failure_message": "",
                        }
                    )
                message = f"[{index}/57] {sample['sample_id']} {summary_rows[-1]['status']} excluded={summary_rows[-1]['excluded_building_count']}"
            except Exception as error:
                summary_rows.append(
                    {
                        **base, "status": "failed", "runtime_seconds": round(time.perf_counter() - started, 6),
                        "failure_message": str(error),
                    }
                )
                message = f"[{index}/57] {sample['sample_id']} FAILED {error}"
            print(message, flush=True)
            log.write(message + "\n")
            log.flush()

        for sample in regression:
            base = {
                "run_id": run_id, "sample_id": sample["sample_id"], "city": sample["city"],
                "block_id": sample["block_id"],
            }
            try:
                block = blocks[sample["block_id"]]
                buildings = fetch_buildings(connection, block)
                if len(buildings) != int(sample["lod1_building_count"]):
                    raise RuntimeError("Frozen regression building count changed")
                old_path = CH03_DIR / sample["baseline_graph_path"]
                old_record = read_graph(old_path)
                outcome = encode_block_to_v031(
                    sample, block, buildings, sample_manifest_sha256=manifest_hash,
                    tessellation_config=DEFAULT_TESSELLATION_CONFIG,
                )
                if outcome.status != "completed" or outcome.record is None:
                    raise RuntimeError(f"Regression encoding status was {outcome.status}")
                old_signature = _structure_signature(old_record)
                new_signature = _structure_signature(outcome.record)
                node_match = old_signature["nodes"] == new_signature["nodes"]
                cell_match = old_signature["cells"] == new_signature["cells"]
                edge_match = old_signature["edges"] == new_signature["edges"]
                regression_rows.append(
                    {
                        **base, "status": "completed", "excluded_building_count": len(outcome.excluded_buildings),
                        "node_signature_match": node_match, "cell_signature_match": cell_match,
                        "edge_signature_match": edge_match, "all_match": node_match and cell_match and edge_match,
                        "baseline_node_count": len(old_signature["nodes"]),
                        "baseline_cell_count": len(old_signature["cells"]),
                        "baseline_edge_count": len(old_signature["edges"]),
                        "new_node_count": len(new_signature["nodes"]),
                        "new_cell_count": len(new_signature["cells"]),
                        "new_edge_count": len(new_signature["edges"]), "failure_message": "",
                    }
                )
            except Exception as error:
                regression_rows.append({**base, "status": "failed", "failure_message": str(error)})
            message = f"[regression] {sample['sample_id']} {regression_rows[-1]['status']} match={regression_rows[-1].get('all_match', '')}"
            print(message, flush=True)
            log.write(message + "\n")

    _write_csv(summary_path, summary_rows, SUMMARY_FIELDS)
    _write_csv(exclusions_path, exclusion_rows, EXCLUSION_FIELDS)
    _write_csv(regression_path, regression_rows, REGRESSION_FIELDS)

    environment = {
        "captured_at": datetime.now().astimezone().isoformat(),
        "experiment_id": "CH3-P3.1-NODE-CLEANUP-V031",
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "libraries": _library_versions(),
        "database": {
            "name": database_name, "server": database_version,
            "access": "read-only transaction through shared/db_readonly.py",
            "candidate_relation": "summary.lod1_candidate_blocks",
        },
    }
    (RESULTS_DIR / "environment.json").write_text(
        json.dumps(environment, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    freeze = {
        "run_id": run_id,
        "sample_manifest_sha256": manifest_hash,
        "affected_manifest_sha256": _sha256(AFFECTED_MANIFEST),
        "regression_manifest_sha256": _sha256(REGRESSION_MANIFEST),
        "source_sha256": {
            name: _sha256(PHASE_DIR / name)
            for name in ("filter_buildings_v031.py", "export_phase3_1.py", "run_phase3_1.py")
        },
        "baseline_paths": {
            "sample_manifest": str(phase3_manifest),
            "database_snapshot": str(BASELINE_RESULTS / "database_snapshot.json"),
            "encoding_status": str(BASELINE_RESULTS / "audit" / "encoding_status.csv"),
            "graph_directory": str(BASELINE_RESULTS / "dataset" / "graphs"),
        },
        "tessellation_parameters": DEFAULT_TESSELLATION_CONFIG.__dict__,
    }
    (RESULTS_DIR / "freeze.json").write_text(
        json.dumps(freeze, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    result = {
        "run_id": run_id,
        "affected_status_count": len(summary_rows),
        "completed_graph_count": sum(row["status"] == "completed" for row in summary_rows),
        "excluded_graph_count": sum(row["status"] == "excluded_no_valid_building" for row in summary_rows),
        "failed_count": sum(row["status"] == "failed" for row in summary_rows),
        "excluded_building_count": len(exclusion_rows),
        "regression_match_count": sum(row.get("all_match") is True for row in regression_rows),
    }
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=("affected",), default="affected")
    parser.parse_args()
    result = run_affected()
    return 0 if result["failed_count"] == 0 and result["regression_match_count"] == 3 else 1


if __name__ == "__main__":
    raise SystemExit(main())
