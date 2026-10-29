"""Model-neutral v0.3 graph encoding and serialization."""

from __future__ import annotations

import copy
import gzip
import io
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np


from .assembly import assemble_graph
from .tessellation import (
    TessellationConfig,build_cell_adjacency,build_morphological_cells,validate_tessellation_graph,
)

ENCODING_VERSION = "0.3-candidate"
DEFAULT_TESSELLATION_CONFIG = TessellationConfig(
    shrink_m=0.4,
    segment_m=0.5,
    min_shared_boundary_m=1e-6,
)


def apply_weak_edge_flags(
    edges: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Add descriptive weak-edge flags while preserving every edge and its order."""
    flagged = copy.deepcopy(edges)
    gaps = np.asarray(
        [float(edge["footprint_gap_block_norm"]) for edge in flagged], dtype=float
    )
    if len(gaps) >= 4:
        q1, q3 = np.quantile(gaps, [0.25, 0.75])
        long_threshold = float(q3 + 1.5 * (q3 - q1))
    else:
        long_threshold = None
    if len(flagged) >= 10:
        shared_threshold = float(
            np.quantile(
                np.asarray(
                    [float(edge["shared_boundary_ratio_min"]) for edge in flagged],
                    dtype=float,
                ),
                0.10,
            )
        )
    else:
        shared_threshold = None

    for edge in flagged:
        edge["long_distance_candidate"] = bool(
            long_threshold is not None
            and float(edge["footprint_gap_block_norm"]) > long_threshold
        )
        edge["low_shared_candidate"] = bool(
            shared_threshold is not None
            and float(edge["shared_boundary_ratio_min"]) <= shared_threshold
        )
        edge["combined_weak_candidate"] = bool(
            edge["long_distance_candidate"] and edge["low_shared_candidate"]
        )

    return flagged, {
        "edge_count": len(flagged),
        "long_distance_rule": "footprint_gap_block_norm > within-graph Tukey upper fence; n>=4",
        "long_distance_threshold": long_threshold,
        "long_distance_candidate_count": sum(edge["long_distance_candidate"] for edge in flagged),
        "low_shared_rule": "shared_boundary_ratio_min <= within-graph 10th percentile; n>=10",
        "low_shared_threshold": shared_threshold,
        "low_shared_candidate_count": sum(edge["low_shared_candidate"] for edge in flagged),
        "combined_weak_candidate_count": sum(edge["combined_weak_candidate"] for edge in flagged),
        "edge_deletion_count": 0,
    }


def _add_node_missing_reasons(nodes: list[dict[str, Any]]) -> None:
    for node in nodes:
        reasons = {}
        for field, missing in node.get("missing", {}).items():
            if not missing:
                reasons[field] = None
            elif field == "orientation":
                reasons[field] = "geometry_orientation_unavailable"
            else:
                reasons[field] = "source_value_null"
        node["missing_reasons"] = reasons


def encode_block_to_v03(
    sample: dict[str, Any],
    block_record: dict[str, Any],
    building_records: list[dict[str, Any]],
    *,
    sample_manifest_sha256: str,
    tessellation_config: TessellationConfig = DEFAULT_TESSELLATION_CONFIG,
) -> dict[str, Any]:
    """Reuse v0.1 nodes and v0.2 tessellation relations in a v0.3 record."""
    source = assemble_graph(sample, block_record, building_records)
    _add_node_missing_reasons(source["nodes"])
    cells, failures, cell_diagnostics = build_morphological_cells(
        block_record["geometry"], source["nodes"], tessellation_config
    )
    raw_edges, adjacency_diagnostics = build_cell_adjacency(
        cells,
        source["nodes"],
        float(block_record["geometry"].area),
        tessellation_config,
    )
    edges, weak_summary = apply_weak_edge_flags(raw_edges)
    graph = dict(source["graph"])
    graph.update(
        {
            "node_count": len(source["nodes"]),
            "cell_count": len(cells),
            "cell_failure_count": len(failures),
            "edge_count": len(edges),
        }
    )
    quality_flags = list(graph.get("quality_flags", []))
    if failures:
        quality_flags.append("cell_failures_present")
    quality_flags.extend(cell_diagnostics.get("warnings", []))
    graph["quality_flags"] = list(dict.fromkeys(quality_flags))

    record = {
        "encoding_version": ENCODING_VERSION,
        "experiment_id": "CH3-P3-ENCODING-SCALE-V03",
        "sample_manifest_sha256": sample_manifest_sha256,
        "sample": dict(sample),
        "parameters": asdict(tessellation_config),
        "graph": graph,
        "block_attributes": source["block_attributes"],
        "nodes": source["nodes"],
        "cells": cells,
        "cell_failures": failures,
        "edges": edges,
        "weak_edge_summary": weak_summary,
        "diagnostics": {
            "v01_node_derivation": source["diagnostics"],
            "tessellation": cell_diagnostics,
            "adjacency": adjacency_diagnostics,
        },
    }
    record["validation"] = validate_v03_record(record)
    return record


def _walk_nonfinite(value: Any, path: str = "") -> list[str]:
    issues: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            issues.extend(_walk_nonfinite(item, f"{path}.{key}" if path else str(key)))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            issues.extend(_walk_nonfinite(item, f"{path}[{index}]"))
    elif isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        issues.append(path)
    return issues


def validate_v03_record(record: dict[str, Any]) -> dict[str, Any]:
    """Report record violations without mutating or repairing the input."""
    issues: list[dict[str, Any]] = []

    def add(issue_type: str, message: str, **details: Any) -> None:
        issues.append({"issue_type": issue_type, "message": message, **details})

    graph = record.get("graph", {})
    collections = {
        "node": record.get("nodes", []),
        "cell": record.get("cells", []),
        "cell_failure": record.get("cell_failures", []),
        "edge": record.get("edges", []),
    }
    for name, items in collections.items():
        expected = graph.get(f"{name}_count")
        if expected != len(items):
            add(
                f"{name}_count_mismatch",
                f"graph.{name}_count={expected!r}, actual={len(items)}",
            )

    node_ids = [str(node["building_id"]) for node in collections["node"]]
    if len(node_ids) != len(set(node_ids)):
        add("duplicate_node_id", "Node building IDs are not unique")
    cell_ids = {str(cell["building_id"]) for cell in collections["cell"]}
    failure_ids = {str(item["building_id"]) for item in collections["cell_failure"]}
    if cell_ids & failure_ids:
        add("cell_and_failure_overlap", "A building has both a cell and a cell failure")
    if cell_ids | failure_ids != set(node_ids):
        add("cell_coverage_mismatch", "Cells plus explicit failures do not cover all nodes")

    tess_validation = validate_tessellation_graph(record)
    for issue in tess_validation["issues"]:
        add(issue["issue_type"], issue["message"], edge_id=issue.get("edge_id"))

    for node in collections["node"]:
        reasons = node.get("missing_reasons", {})
        for field, missing in node.get("missing", {}).items():
            if missing and not reasons.get(field):
                add("missing_reason_absent", f"Node {node['building_id']} field {field} lacks a reason")
    for edge in collections["edge"]:
        for field, missing in edge.get("missing", {}).items():
            if missing and not edge.get("missing_reasons", {}).get(field):
                add("missing_reason_absent", f"Edge {edge.get('edge_id')} field {field} lacks a reason")

    nonfinite_paths = _walk_nonfinite(record)
    if nonfinite_paths:
        add("nonfinite_numeric", "Record contains non-finite numeric values", paths=nonfinite_paths)

    weak = record.get("weak_edge_summary", {})
    for field, edge_field in (
        ("long_distance_candidate_count", "long_distance_candidate"),
        ("low_shared_candidate_count", "low_shared_candidate"),
        ("combined_weak_candidate_count", "combined_weak_candidate"),
    ):
        actual = sum(bool(edge.get(edge_field)) for edge in collections["edge"])
        if weak.get(field) != actual:
            add("weak_edge_count_mismatch", f"{field}={weak.get(field)!r}, actual={actual}")

    return {
        "passed": not issues,
        "issue_count": len(issues),
        "issue_types": sorted({issue["issue_type"] for issue in issues}),
        "issues": issues,
    }


def write_graph(record: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(record, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
    if path.suffix == ".gz":
        with path.open("wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
                compressed.write(payload)
    else:
        path.write_bytes(payload)


def read_graph(path: Path) -> dict[str, Any]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    return json.loads(path.read_text(encoding="utf-8"))
