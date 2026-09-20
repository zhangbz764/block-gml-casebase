"""Rebuild the 15 frozen Chapter 3 phase 2.5 graph outputs."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from shapely import wkt

from tessellation_encoding import (
    TessellationConfig,
    build_cell_adjacency,
    build_morphological_cells,
    read_json,
    validate_tessellation_graph,
    write_json,
)


PHASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PHASE_DIR / "results"
REPO_ROOT = PHASE_DIR.parents[2]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def edge_key(edge: dict[str, Any]) -> tuple[str, str]:
    return tuple(sorted((str(edge["source"]), str(edge["target"]))))


def scope_class(sample_id: str) -> str:
    if sample_id in {"contrast_low_coverage", "contrast_orientation_irregular"}:
        return "out_of_scope_candidate"
    if sample_id == "contrast_high_coverage":
        return "data_quality_stress"
    return "core_or_boundary_case"


def text_flags(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    return [item for item in str(value).split(";") if item]


def structural_signature(graph: dict[str, Any]) -> dict[str, list[str]]:
    return {
        "node_ids": [str(item["building_id"]) for item in graph["nodes"]],
        "cell_ids": [str(item["building_id"]) for item in graph["cells"]],
        "edge_ids": [str(item["edge_id"]) for item in graph["edges"]],
    }


def corrected_graph_metadata(
    source_graph: dict[str, Any],
    nodes: list[dict[str, Any]],
    cells: list[dict[str, Any]],
    cell_failures: list[dict[str, Any]],
    edges: list[dict[str, Any]],
) -> dict[str, Any]:
    """Copy v0.1 block metadata while replacing v0.2 structure counts."""
    metadata = dict(source_graph)
    metadata.update(
        {
            "node_count": len(nodes),
            "cell_count": len(cells),
            "cell_failure_count": len(cell_failures),
            "edge_count": len(edges),
        }
    )
    return metadata


def nonfinite_numeric_count(edges: list[dict[str, Any]]) -> int:
    count = 0
    for edge in edges:
        for key, value in edge.items():
            if key in {"missing", "missing_reasons"} or value is None:
                continue
            if isinstance(value, (int, float, np.integer, np.floating)) and not math.isfinite(float(value)):
                count += 1
    return count


def main() -> int:
    run_started = time.perf_counter()
    parameters = read_json(PHASE_DIR / "tessellation_parameters.json")
    config = TessellationConfig(
        shrink_m=float(parameters["shrink_m"]),
        segment_m=float(parameters["segment_m"]),
        min_shared_boundary_m=float(parameters["min_shared_boundary_m"]),
    )
    with (RESULTS_DIR / "input_manifest.csv").open(encoding="utf-8-sig", newline="") as handle:
        input_rows = list(csv.DictReader(handle))

    sample_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    issue_rows: list[dict[str, Any]] = []
    failed_samples: list[str] = []
    (RESULTS_DIR / "cells").mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "graphs").mkdir(parents=True, exist_ok=True)

    for manifest in input_rows:
        sample_started = time.perf_counter()
        sample_id = manifest["sample_id"]
        classification = scope_class(sample_id)
        source_path = REPO_ROOT / manifest["source_path"]
        try:
            actual_hash = sha256_file(source_path)
            if actual_hash != manifest["sha256"]:
                raise ValueError(
                    f"source SHA-256 mismatch: expected {manifest['sha256']}, got {actual_hash}"
                )
            source = read_json(source_path)
            block = wkt.loads(source["graph"]["block_geometry_wkt_metric"])
            cells, cell_failures, cell_diagnostics = build_morphological_cells(
                block, source["nodes"], config
            )
            edges, adjacency_diagnostics = build_cell_adjacency(
                cells, source["nodes"], float(block.area), config
            )
            validation = validate_tessellation_graph(
                {"nodes": source["nodes"], "cells": cells, "edges": edges}
            )

            v01_edges = {edge_key(edge) for edge in source["edges"]}
            v02_edges = {edge_key(edge) for edge in edges}
            common_edges = v01_edges & v02_edges
            added_edges = v02_edges - v01_edges
            removed_edges = v01_edges - v02_edges

            normalized_gaps = np.asarray(
                [float(edge["footprint_gap_block_norm"]) for edge in edges], dtype=float
            )
            if len(normalized_gaps) >= 4:
                q1, q3 = np.quantile(normalized_gaps, [0.25, 0.75])
                long_edge_threshold = float(q3 + 1.5 * (q3 - q1))
                long_edges = [
                    edge
                    for edge in edges
                    if float(edge["footprint_gap_block_norm"]) > long_edge_threshold
                ]
            else:
                long_edge_threshold = None
                long_edges = []

            source_quality_flags = text_flags(source["graph"].get("quality_flags"))
            warnings = list(cell_diagnostics.get("warnings", []))
            if cell_failures:
                warnings.append("cell_failures_present")
            if validation["issues"]:
                warnings.append("graph_validation_issues")
            if long_edges:
                warnings.append("long_distance_candidates")
            warnings = list(dict.fromkeys(warnings + source_quality_flags))

            graph_output = {
                "encoding_version": "0.2",
                "source_v01": {
                    "path": manifest["source_path"],
                    "sha256": actual_hash,
                    "encoding_version": source.get("encoding_version"),
                },
                "sample": source["sample"],
                "scope_class": classification,
                "parameters": {
                    "shrink_m": config.shrink_m,
                    "segment_m": config.segment_m,
                    "min_shared_boundary_m": config.min_shared_boundary_m,
                },
                "graph": corrected_graph_metadata(
                    source["graph"],
                    source["nodes"],
                    cells,
                    cell_failures,
                    edges,
                ),
                "block_attributes": source.get("block_attributes", {}),
                "nodes": source["nodes"],
                "cells": cells,
                "cell_failures": cell_failures,
                "edges": edges,
                "diagnostics": {
                    "tessellation": cell_diagnostics,
                    "adjacency": adjacency_diagnostics,
                    "validation": validation,
                    "v01_v02_edge_comparison": {
                        "v01_edge_count": len(v01_edges),
                        "v02_edge_count": len(v02_edges),
                        "common_edge_count": len(common_edges),
                        "added_edge_count": len(added_edges),
                        "removed_edge_count": len(removed_edges),
                    },
                    "long_distance_candidate": {
                        "rule": "Tukey upper fence on footprint_gap_block_norm",
                        "threshold": long_edge_threshold,
                        "edge_count": len(long_edges),
                    },
                },
                "quality_flags": warnings,
            }
            cell_output = {
                "encoding_version": "0.2",
                "sample_id": sample_id,
                "block_id": source["graph"]["block_id"],
                "source_v01_sha256": actual_hash,
                "parameters": graph_output["parameters"],
                "block_geometry_wkt_metric": source["graph"]["block_geometry_wkt_metric"],
                "cells": cells,
                "failures": cell_failures,
                "diagnostics": cell_diagnostics,
            }
            cell_path = RESULTS_DIR / "cells" / f"{sample_id}.cells.json"
            graph_path = RESULTS_DIR / "graphs" / f"{sample_id}.graph.json"
            write_json(cell_output, cell_path)
            write_json(graph_output, graph_path)
            reloaded = read_json(graph_path)
            roundtrip_ok = structural_signature(graph_output) == structural_signature(reloaded)

            for failure in cell_failures:
                issue_rows.append(
                    {
                        "sample_id": sample_id,
                        "issue_level": "building",
                        "issue_type": failure["failure_type"],
                        "building_id": failure["building_id"],
                        "edge_id": "",
                        "message": failure["message"],
                        "value": "",
                        "criterion": "cell construction",
                    }
                )
            for issue in validation["issues"]:
                issue_rows.append(
                    {
                        "sample_id": sample_id,
                        "issue_level": "edge_or_graph",
                        "issue_type": issue["issue_type"],
                        "building_id": "",
                        "edge_id": issue.get("edge_id", ""),
                        "message": issue["message"],
                        "value": "",
                        "criterion": "graph invariant",
                    }
                )
            for edge in long_edges:
                issue_rows.append(
                    {
                        "sample_id": sample_id,
                        "issue_level": "edge",
                        "issue_type": "long_distance_candidate",
                        "building_id": "",
                        "edge_id": edge["edge_id"],
                        "message": "Review aid only; edge is retained",
                        "value": edge["footprint_gap_block_norm"],
                        "criterion": f"> Tukey upper fence {long_edge_threshold}",
                    }
                )
            if classification != "core_or_boundary_case":
                issue_rows.append(
                    {
                        "sample_id": sample_id,
                        "issue_level": "sample",
                        "issue_type": classification,
                        "building_id": "",
                        "edge_id": "",
                        "message": "Preserved for separate interpretation, not counted as a normal-case pass",
                        "value": "",
                        "criterion": "task specification",
                    }
                )
            for flag in source_quality_flags:
                issue_rows.append(
                    {
                        "sample_id": sample_id,
                        "issue_level": "sample",
                        "issue_type": "source_quality_flag",
                        "building_id": "",
                        "edge_id": "",
                        "message": flag,
                        "value": "",
                        "criterion": "v0.1 provenance",
                    }
                )

            elapsed = time.perf_counter() - sample_started
            failure_counts = Counter(item["failure_type"] for item in cell_failures)
            sample_rows.append(
                {
                    "sample_id": sample_id,
                    "sample_category": manifest["sample_category"],
                    "sample_subtype": manifest["sample_subtype"],
                    "scope_class": classification,
                    "city": manifest["city"],
                    "block_id": manifest["block_id"],
                    "node_count": len(source["nodes"]),
                    "cell_count": len(cells),
                    "failed_cell_count": len(cell_failures),
                    "cell_failure_types": json.dumps(failure_counts, ensure_ascii=False, sort_keys=True),
                    "v01_edge_count": len(v01_edges),
                    "v02_edge_count": len(v02_edges),
                    "runtime_seconds": elapsed,
                    "status": "completed_with_cell_failures" if cell_failures else "completed",
                    "warnings": ";".join(warnings),
                }
            )
            issue_counts = validation["issue_counts"]
            validation_rows.append(
                {
                    "sample_id": sample_id,
                    "scope_class": classification,
                    "input_building_count": len(source["nodes"]),
                    "success_cell_count": len(cells),
                    "failed_cell_count": len(cell_failures),
                    "failure_types": json.dumps(failure_counts, ensure_ascii=False, sort_keys=True),
                    "cell_outside_area_m2": cell_diagnostics["cell_outside_area_m2"],
                    "cell_overlap_area_m2": cell_diagnostics["cell_overlap_area_m2"],
                    "uncovered_area_m2": cell_diagnostics["uncovered_area_m2"],
                    "uncovered_area_ratio": cell_diagnostics["uncovered_area_ratio"],
                    "self_loop_count": issue_counts.get("self_loop", 0),
                    "duplicate_edge_count": issue_counts.get("duplicate_edge_id", 0),
                    "dangling_endpoint_count": issue_counts.get("dangling_edge", 0),
                    "nonfinite_numeric_count": nonfinite_numeric_count(edges),
                    "min_shared_boundary_m": min(
                        (edge["shared_cell_boundary_m"] for edge in edges), default=None
                    ),
                    "max_footprint_gap_m": max(
                        (edge["footprint_gap_m"] for edge in edges), default=None
                    ),
                    "long_distance_candidate_count": len(long_edges),
                    "v01_v02_common_edge_count": len(common_edges),
                    "v02_added_edge_count": len(added_edges),
                    "v02_removed_edge_count": len(removed_edges),
                    "graph_roundtrip_ok": roundtrip_ok,
                    "runtime_seconds": elapsed,
                    "sample_point_count": cell_diagnostics["sample_point_count"],
                    "validation_passed": validation["passed"] and roundtrip_ok,
                }
            )
            print(
                f"[{sample_id}] nodes={len(source['nodes'])} cells={len(cells)} "
                f"edges={len(edges)} failures={len(cell_failures)} seconds={elapsed:.2f}",
                flush=True,
            )
        except Exception as error:
            failed_samples.append(sample_id)
            elapsed = time.perf_counter() - sample_started
            sample_rows.append(
                {
                    "sample_id": sample_id,
                    "sample_category": manifest["sample_category"],
                    "sample_subtype": manifest["sample_subtype"],
                    "scope_class": classification,
                    "city": manifest["city"],
                    "block_id": manifest["block_id"],
                    "node_count": manifest["node_count"],
                    "cell_count": 0,
                    "failed_cell_count": manifest["node_count"],
                    "cell_failure_types": json.dumps({"sample_execution_failure": 1}),
                    "v01_edge_count": manifest["v01_edge_count"],
                    "v02_edge_count": 0,
                    "runtime_seconds": elapsed,
                    "status": "failed",
                    "warnings": "sample_execution_failure",
                }
            )
            issue_rows.append(
                {
                    "sample_id": sample_id,
                    "issue_level": "sample",
                    "issue_type": "sample_execution_failure",
                    "building_id": "",
                    "edge_id": "",
                    "message": str(error),
                    "value": "",
                    "criterion": "execution",
                }
            )
            print(f"[{sample_id}] FAILED: {error}", flush=True)

    sample_fields = [
        "sample_id", "sample_category", "sample_subtype", "scope_class", "city",
        "block_id", "node_count", "cell_count", "failed_cell_count",
        "cell_failure_types", "v01_edge_count", "v02_edge_count",
        "runtime_seconds", "status", "warnings",
    ]
    validation_fields = [
        "sample_id", "scope_class", "input_building_count", "success_cell_count",
        "failed_cell_count", "failure_types", "cell_outside_area_m2",
        "cell_overlap_area_m2", "uncovered_area_m2", "uncovered_area_ratio",
        "self_loop_count", "duplicate_edge_count", "dangling_endpoint_count",
        "nonfinite_numeric_count", "min_shared_boundary_m", "max_footprint_gap_m",
        "long_distance_candidate_count", "v01_v02_common_edge_count",
        "v02_added_edge_count", "v02_removed_edge_count", "graph_roundtrip_ok",
        "runtime_seconds", "sample_point_count", "validation_passed",
    ]
    issue_fields = [
        "sample_id", "issue_level", "issue_type", "building_id", "edge_id",
        "message", "value", "criterion",
    ]
    write_csv(RESULTS_DIR / "sample_manifest.csv", sample_rows, sample_fields)
    write_csv(RESULTS_DIR / "validation_summary.csv", validation_rows, validation_fields)
    write_csv(RESULTS_DIR / "validation_issues.csv", issue_rows, issue_fields)

    print(
        json.dumps(
            {
                "samples": len(input_rows),
                "failed_samples": failed_samples,
                "total_runtime_seconds": round(time.perf_counter() - run_started, 3),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 1 if failed_samples else 0


if __name__ == "__main__":
    raise SystemExit(main())
