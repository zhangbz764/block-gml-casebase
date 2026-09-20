"""Feature, structure, quality, and resource audit for phase 3 graphs."""

from __future__ import annotations

import csv
import gzip
import io
import itertools
import json
import math
from collections import Counter, deque
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.stats import spearmanr

from export_phase3 import read_graph


PHASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PHASE_DIR / "results"
DATASET_DIR = RESULTS_DIR / "dataset"
AUDIT_DIR = RESULTS_DIR / "audit"


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool) and math.isfinite(float(value))


def numeric_profile(values: Iterable[Any]) -> dict[str, Any]:
    values = list(values)
    valid = np.asarray([float(value) for value in values if _finite(value)], dtype=float)
    result = {
        "record_count": len(values),
        "valid_count": int(len(valid)),
        "missing_count": len(values) - int(len(valid)),
        "missing_ratio": (len(values) - len(valid)) / len(values) if values else None,
        "zero_count": int(np.sum(valid == 0)) if len(valid) else 0,
        "zero_ratio_valid": float(np.mean(valid == 0)) if len(valid) else None,
    }
    if len(valid):
        quantiles = np.quantile(valid, [0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1])
        result.update(
            dict(zip(("min", "p01", "p05", "p25", "p50", "p75", "p95", "p99", "max"), map(float, quantiles)))
        )
        result["mean"] = float(np.mean(valid))
        result["std"] = float(np.std(valid))
        median = float(np.median(valid))
        result["right_skew_hint"] = bool(result["p99"] > max(1e-12, abs(median)) * 10 and result["min"] >= 0)
        result["suggested_transform"] = "log1p_review" if result["right_skew_hint"] else "none"
    else:
        result.update({name: None for name in ("min", "p01", "p05", "p25", "p50", "p75", "p95", "p99", "max", "mean", "std")})
        result["right_skew_hint"] = False
        result["suggested_transform"] = "not_applicable"
    return result


def spearman_pairwise(rows: Iterable[dict[str, Any]], first: str, second: str) -> dict[str, Any]:
    pairs = [
        (float(row[first]), float(row[second]))
        for row in rows
        if _finite(row.get(first)) and _finite(row.get(second))
    ]
    if len(pairs) < 2:
        rho = None
    else:
        rho_value = float(spearmanr([item[0] for item in pairs], [item[1] for item in pairs]).statistic)
        rho = rho_value if math.isfinite(rho_value) else None
    return {"first_field": first, "second_field": second, "valid_pair_count": len(pairs), "spearman_rho": rho}


def build_feature_registry() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    def add(entity: str, fields: Iterable[str], role: str, unit: str, default: str, rationale: str, transform: str = "none", missing: str = "retain null and mask") -> None:
        for field in fields:
            rows.append(
                {
                    "entity": entity,
                    "field": f"{entity}.{field}",
                    "role": role,
                    "unit": unit,
                    "transform_or_encoding": transform,
                    "missing_rule": missing,
                    "default_input": default,
                    "rationale": rationale,
                }
            )

    add("graph", ["sample_id", "block_id", "city", "country", "sample_groups", "storage_srid", "metric_srid", "source_block_relation", "source_building_relation", "block_geometry_wkt_metric"], "provenance_trace", "identifier", "no", "reproduction or source lookup", missing="not allowed")
    add("graph", ["node_count", "cell_count", "edge_count", "connected_components", "isolated_node_count", "average_degree"], "auxiliary_interpretation", "count_or_ratio", "no", "graph size and interpretation; not default morphology input")
    add("graph", ["cell_failure_count", "quality_flags", "long_distance_candidate_count", "low_shared_candidate_count", "combined_weak_candidate_count"], "quality_audit", "count_or_flag", "no", "quality screening only")

    add("block", ["elongation", "compactness", "bcr", "convexity", "rectangularity", "far", "bcd", "osr", "avh_w", "hcv", "footprint_cv", "avg_complexity", "avg_rectangularity", "avg_convexity", "orientation_entropy"], "core_model", "ratio_or_index", "yes", "scale-reduced block morphology candidate")
    add("block", ["area_m2", "lod1_building_count", "avh", "hsd", "h_max", "h_range", "total_volume", "avg_volume", "avg_footprint", "avg_orientation", "avg_nn_distance"], "auxiliary_interpretation", "source_unit", "no", "absolute scale or duplicate-friendly interpretation field")

    add("node", ["building_id", "citygml_id", "representative_point_x_m", "representative_point_y_m", "geometry_wkt_metric", "source_index"], "provenance_trace", "identifier_or_projected_m", "no", "traceability or absolute projected location")
    add("node", ["footprint_compactness", "footprint_rectangularity", "footprint_convexity", "orientation_sin_2theta", "orientation_cos_2theta", "local_x_normalized", "local_y_normalized", "boundary_distance_normalized"], "core_model", "normalized_or_ratio", "yes", "normalized position and footprint form")
    add("node", ["height_m", "source_area_m2", "source_perimeter_m", "footprint_area_m2", "footprint_perimeter_m", "orientation_deg", "local_x_m", "local_y_m", "boundary_distance_m"], "auxiliary_interpretation", "m_or_m2_or_degree", "no", "absolute or human-readable counterpart")
    add("node", ["missing", "missing_reasons", "quality"], "quality_audit", "mask_or_flag", "no", "missingness and geometry quality")

    add("cell", ["building_id", "geometry_wkt_metric"], "provenance_trace", "identifier_or_wkt", "no", "cell-to-building traceability")
    add("cell", ["area_m2", "perimeter_m"], "auxiliary_interpretation", "m2_or_m", "no", "absolute cell size for interpretation")
    add("cell", ["geometry_valid"], "quality_audit", "flag", "no", "geometry check")

    add("edge", ["edge_id", "source", "target"], "provenance_trace", "identifier", "no", "stable edge lookup")
    add("edge", ["shared_boundary_ratio_min", "shared_boundary_ratio_mean", "footprint_gap_block_norm", "centroid_distance_block_norm", "orientation_similarity", "long_side_overlap_ratio", "long_side_offset_norm", "connection_axis_alignment_i", "connection_axis_alignment_j", "log_area_ratio_abs", "log_height_ratio_abs", "volume_ratio"], "core_model", "normalized_ratio_or_degree", "yes", "relative relation candidate")
    add("edge", ["shared_cell_boundary_m", "footprint_gap_m", "centroid_distance_m", "orientation_diff_deg", "height_diff_m"], "auxiliary_interpretation", "m_or_degree", "no", "absolute or human-readable counterpart")
    add("edge", ["missing", "missing_reasons", "long_distance_candidate", "low_shared_candidate", "combined_weak_candidate"], "quality_audit", "mask_or_flag", "no", "review flags never delete edges")
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = fieldnames or (list(rows[0]) if rows else [])
    if path.suffix == ".gz":
        raw = path.open("wb")
        compressed = gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0)
        handle = io.TextIOWrapper(compressed, encoding="utf-8-sig", newline="")
    else:
        handle = path.open("w", encoding="utf-8-sig", newline="")
    with handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _graph_connectivity(node_ids: list[str], edges: list[dict[str, Any]]) -> tuple[int, int, float]:
    if not node_ids:
        return 0, 0, 0.0
    adjacency = {node_id: set() for node_id in node_ids}
    for edge in edges:
        adjacency[str(edge["source"])].add(str(edge["target"]))
        adjacency[str(edge["target"])].add(str(edge["source"]))
    unseen = set(node_ids)
    components = 0
    while unseen:
        components += 1
        queue = deque([unseen.pop()])
        while queue:
            current = queue.popleft()
            for neighbor in adjacency[current]:
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    queue.append(neighbor)
    isolated = sum(not neighbors for neighbors in adjacency.values())
    return components, isolated, 2 * len(edges) / len(node_ids)


def _simple_values(item: dict[str, Any], excluded: set[str]) -> dict[str, Any]:
    return {key: value for key, value in item.items() if key not in excluded and not isinstance(value, (dict, list))}


def _profiles(entity_rows: dict[str, list[dict[str, Any]]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    global_rows: list[dict[str, Any]] = []
    city_rows: list[dict[str, Any]] = []
    for entity, rows in entity_rows.items():
        numeric_fields = sorted({key for row in rows for key, value in row.items() if _finite(value) and key not in {"city"}})
        for field in numeric_fields:
            global_rows.append({"entity": entity, "field": f"{entity}.{field}", **numeric_profile(row.get(field) for row in rows)})
        for city in sorted({str(row["city"]) for row in rows}):
            subset = [row for row in rows if row["city"] == city]
            for field in numeric_fields:
                city_rows.append({"entity": entity, "city": city, "field": f"{entity}.{field}", **numeric_profile(row.get(field) for row in subset)})
    return global_rows, city_rows


def _redundancy(entity_rows: dict[str, list[dict[str, Any]]], threshold: float = 0.90) -> list[dict[str, Any]]:
    output = []
    for entity, rows in entity_rows.items():
        fields = sorted({key for row in rows for key, value in row.items() if _finite(value) and key not in {"city"}})
        for first, second in itertools.combinations(fields, 2):
            result = spearman_pairwise(rows, first, second)
            rho = result["spearman_rho"]
            if result["valid_pair_count"] >= 20 and rho is not None and abs(rho) >= threshold:
                output.append(
                    {
                        "entity": entity,
                        "first_field": f"{entity}.{first}",
                        "second_field": f"{entity}.{second}",
                        "valid_pair_count": result["valid_pair_count"],
                        "spearman_rho": rho,
                        "absolute_rho": abs(rho),
                        "action": "manual_review_only",
                    }
                )
    return sorted(output, key=lambda row: (-row["absolute_rho"], row["entity"], row["first_field"], row["second_field"]))


def audit_dataset() -> dict[str, Any]:
    with (AUDIT_DIR / "encoding_status.csv").open(encoding="utf-8-sig", newline="") as handle:
        status_rows = list(csv.DictReader(handle))
    completed = [row for row in status_rows if row["status"] == "completed"]
    graph_rows: list[dict[str, Any]] = []
    node_rows: list[dict[str, Any]] = []
    cell_rows: list[dict[str, Any]] = []
    edge_rows: list[dict[str, Any]] = []
    structure_rows: list[dict[str, Any]] = []
    block_rows: list[dict[str, Any]] = []
    quality_counts: Counter[tuple[str, str]] = Counter()
    for status in status_rows:
        if status["status"] == "failed":
            failure_type = (
                "encoding_failure:invalid_building_footprint"
                if "Invalid footprint" in status.get("failure_message", "")
                else "encoding_failure:other"
            )
            quality_counts[(status["city"], failure_type)] += 1

    for status in completed:
        record = read_graph(DATASET_DIR / "graphs" / f"{status['sample_id']}.graph.json.gz")
        city = str(record["graph"]["city"])
        sample_id = str(status["sample_id"])
        nodes = record["nodes"]
        edges = record["edges"]
        components, isolated, average_degree = _graph_connectivity(
            [str(node["building_id"]) for node in nodes], edges
        )
        weak = record["weak_edge_summary"]
        graph_row = {
            "sample_id": sample_id,
            "block_id": record["graph"]["block_id"],
            "city": city,
            "country": record["graph"]["country"],
            "sample_groups": status["sample_groups"],
            "storage_srid": record["graph"]["storage_srid"],
            "metric_srid": record["graph"]["metric_srid"],
            "node_count": len(nodes),
            "cell_count": len(record["cells"]),
            "cell_failure_count": len(record["cell_failures"]),
            "edge_count": len(edges),
            "connected_components": components,
            "isolated_node_count": isolated,
            "average_degree": average_degree,
            "long_distance_candidate_count": weak["long_distance_candidate_count"],
            "low_shared_candidate_count": weak["low_shared_candidate_count"],
            "combined_weak_candidate_count": weak["combined_weak_candidate_count"],
            "long_distance_candidate_ratio": weak["long_distance_candidate_count"] / len(edges) if edges else 0.0,
            "low_shared_candidate_ratio": weak["low_shared_candidate_count"] / len(edges) if edges else 0.0,
            "combined_weak_candidate_ratio": weak["combined_weak_candidate_count"] / len(edges) if edges else 0.0,
        }
        for field, value in record["block_attributes"]["values"].items():
            graph_row[field] = value
        block_rows.append(
            {
                "sample_id": sample_id,
                "city": city,
                **record["block_attributes"]["values"],
            }
        )
        graph_rows.append(graph_row)
        structure_rows.append(dict(graph_row))

        for flag in record["graph"].get("quality_flags", []):
            quality_counts[(city, str(flag))] += 1
        for failure in record["cell_failures"]:
            quality_counts[(city, f"cell_failure:{failure['failure_type']}")] += 1

        for node in nodes:
            row = {"sample_id": sample_id, "city": city, **_simple_values(node, {"geometry_wkt_metric", "source_index", "missing", "missing_reasons", "quality"})}
            row.update({f"missing_{key}": value for key, value in node.get("missing", {}).items()})
            row.update({f"quality_{key}": value for key, value in node.get("quality", {}).items()})
            node_rows.append(row)
        for cell in record["cells"]:
            cell_rows.append({"sample_id": sample_id, "city": city, **_simple_values(cell, {"geometry_wkt_metric"})})
        for edge in edges:
            row = {"sample_id": sample_id, "city": city, **_simple_values(edge, {"missing", "missing_reasons"})}
            row.update({f"missing_{key}": value for key, value in edge.get("missing", {}).items()})
            edge_rows.append(row)

    _write_csv(DATASET_DIR / "graph_features.csv", graph_rows)
    _write_csv(DATASET_DIR / "node_features.csv.gz", node_rows)
    _write_csv(DATASET_DIR / "cell_features.csv.gz", cell_rows)
    _write_csv(DATASET_DIR / "edge_features.csv.gz", edge_rows)
    _write_csv(AUDIT_DIR / "graph_structure_summary.csv", structure_rows)

    quality_rows = [
        {"city": city, "issue_type": issue, "occurrence_count": count}
        for (city, issue), count in sorted(quality_counts.items())
    ]
    _write_csv(AUDIT_DIR / "quality_issue_summary.csv", quality_rows, ["city", "issue_type", "occurrence_count"])
    resource_rows = [
        {
            "sample_id": row["sample_id"], "city": row["city"],
            "status": row["status"],
            "node_count": row["node_count"], "edge_count": row["edge_count"],
            "sample_point_count": row["sample_point_count"],
            "runtime_seconds": row["runtime_seconds"], "output_bytes": row["output_bytes"],
            "python_peak_memory_mb": row["python_peak_memory_mb"],
        }
        for row in status_rows
    ]
    _write_csv(AUDIT_DIR / "runtime_storage_summary.csv", resource_rows)

    block_metric_fields = set(block_rows[0]) - {"sample_id", "city"} if block_rows else set()
    graph_audit_rows = [
        {key: value for key, value in row.items() if key not in block_metric_fields}
        for row in graph_rows
    ]
    entity_rows = {
        "graph": graph_audit_rows,
        "block": block_rows,
        "node": node_rows,
        "cell": cell_rows,
        "edge": edge_rows,
    }
    global_profiles, city_profiles = _profiles(entity_rows)
    _write_csv(AUDIT_DIR / "feature_profile_global.csv", global_profiles)
    _write_csv(AUDIT_DIR / "feature_profile_by_city.csv", city_profiles)
    redundancy = _redundancy(entity_rows)
    _write_csv(AUDIT_DIR / "feature_redundancy_pairs.csv", redundancy, ["entity", "first_field", "second_field", "valid_pair_count", "spearman_rho", "absolute_rho", "action"])

    registry = build_feature_registry()
    _write_csv(RESULTS_DIR / "feature_role_registry_v0.3.csv", registry)
    summary = {
        "planned": len(status_rows),
        "completed": len(completed),
        "failed": len(status_rows) - len(completed),
        "graph_rows": len(graph_rows),
        "node_rows": len(node_rows),
        "cell_rows": len(cell_rows),
        "edge_rows": len(edge_rows),
        "quality_issue_types": len({issue for _, issue in quality_counts}),
        "redundancy_review_pairs": len(redundancy),
    }
    (AUDIT_DIR / "audit_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return summary


if __name__ == "__main__":
    audit_dataset()
