"""Read-only structural checks and streaming completeness statistics for one Graph."""

from __future__ import annotations

from collections import Counter, defaultdict
import math
from pathlib import Path
from typing import Any

from registry import load_registry


PHASE_DIR = Path(__file__).resolve().parent
META_KEYS = (
    "source_manifest_sha256", "full_manifest_sha256", "part_manifest_sha256",
    "encoding_spec_sha256", "encoding_version", "code_bundle_sha256", "part_id",
    "run_id", "parameter_sha256",
)


def graph_connectivity(node_ids: list[str], edges: list[dict]) -> tuple[int, int, float]:
    if not node_ids:
        return 0, 0, math.nan
    adjacency = {str(node): set() for node in node_ids}
    for edge in edges:
        source, target = str(edge.get("source")), str(edge.get("target"))
        if source in adjacency and target in adjacency and source != target:
            adjacency[source].add(target)
            adjacency[target].add(source)
    isolated = sum(not neighbors for neighbors in adjacency.values())
    components = 0
    unseen = set(adjacency)
    while unseen:
        components += 1
        stack = [unseen.pop()]
        while stack:
            for neighbor in adjacency[stack.pop()]:
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    stack.append(neighbor)
    return components, isolated, sum(map(len, adjacency.values())) / len(adjacency)


def _nonfinite_paths(value: Any, path: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            found.extend(_nonfinite_paths(item, f"{path}.{key}" if path else str(key)))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(_nonfinite_paths(item, f"{path}[{index}]"))
    elif isinstance(value, float) and not math.isfinite(value):
        found.append(path)
    return found


def audit_graph_record(record: dict, status_row: dict, expected_meta: dict) -> tuple[dict, list[dict]]:
    sample = record.get("sample", {})
    identity = {
        "sample_id": str(status_row.get("sample_id") or sample.get("sample_id", "")),
        "city": str(status_row.get("city") or sample.get("city", "")),
        "block_id": str(status_row.get("block_id") or sample.get("block_id", "")),
        "part_id": str(status_row.get("part_id") or expected_meta.get("part_id", "")),
    }
    issues: list[dict] = []

    def issue(kind: str, message: str) -> None:
        issues.append({**identity, "issue_type": kind, "message": message})

    if status_row.get("status") != "completed":
        issue("status_not_completed", f"Expected completed status, got {status_row.get('status')!r}")
    if record.get("encoding_version") != "0.3.2":
        issue("encoding_version_mismatch", "Top-level encoding_version is not 0.3.2")
    if record.get("validation", {}).get("passed") is not True:
        issue("validation_failed", "validation.passed is not true")
    actual_meta = record.get("phase4b", {})
    for key in META_KEYS:
        if actual_meta.get(key) != expected_meta.get(key):
            issue("metadata_mismatch", f"phase4b metadata differs for {key}")

    graph = record.get("graph", {})
    for key in ("sample_id", "city", "block_id"):
        if str(sample.get(key, "")) != identity[key]:
            issue("identity_mismatch", f"sample.{key} differs from the status ledger")
    for key in ("city", "block_id"):
        if str(graph.get(key, "")) != identity[key]:
            issue("identity_mismatch", f"graph.{key} differs from the status ledger")
    nodes = record.get("nodes", [])
    cells = record.get("cells", [])
    failures = record.get("cell_failures", [])
    edges = record.get("edges", [])
    node_ids = [str(item.get("building_id")) for item in nodes]
    cell_ids = [str(item.get("building_id")) for item in cells]
    failure_ids = [str(item.get("building_id")) for item in failures]
    if len(node_ids) != len(set(node_ids)):
        issue("duplicate_node_id", "Node building IDs are not unique")
    if len(cell_ids) != len(set(cell_ids)):
        issue("duplicate_cell_id", "Cell building IDs are not unique")
    if len(failure_ids) != len(set(failure_ids)):
        issue("duplicate_cell_failure_id", "Cell failure building IDs are not unique")
    node_id_set, cell_id_set, failure_id_set = set(node_ids), set(cell_ids), set(failure_ids)
    if cell_id_set & failure_id_set:
        issue("cell_failure_overlap", "A building has both a Cell and a cell failure")
    if failure_id_set - node_id_set:
        issue("unknown_cell_failure_id", f"Cell failures reference unknown Node IDs: {sorted(failure_id_set - node_id_set)}")
    if cell_id_set | failure_id_set != node_id_set:
        issue("cell_coverage_mismatch", "Cells plus cell failures do not explain every Node exactly once")
    edge_ids: set[str] = set()
    pairs: set[tuple[str, str]] = set()
    known = set(node_ids)
    for edge in edges:
        edge_id = str(edge.get("edge_id"))
        source, target = str(edge.get("source")), str(edge.get("target"))
        if edge_id in edge_ids:
            issue("duplicate_edge_id", f"Duplicate edge_id {edge_id}")
        edge_ids.add(edge_id)
        if source == target:
            issue("self_loop", f"Self-loop at {source}")
        pair = tuple(sorted((source, target)))
        if pair in pairs:
            issue("duplicate_undirected_edge", f"Duplicate undirected pair {pair}")
        pairs.add(pair)
        if source not in known or target not in known:
            issue("unknown_edge_endpoint", f"Unknown endpoint in edge {edge_id}")
    for name, items in (("node", nodes), ("cell", cells), ("cell_failure", failures), ("edge", edges)):
        if graph.get(f"{name}_count") != len(items):
            issue(f"{name}_count_mismatch", f"graph.{name}_count differs from list length")
    source_count = graph.get("source_building_count")
    encoded_count = graph.get("encoded_building_count", len(nodes))
    excluded_count = graph.get("excluded_building_count", 0)
    if not isinstance(source_count, (int, float)) or source_count <= 0:
        issue("invalid_source_building_count", "source_building_count must be positive")
    elif source_count != encoded_count + excluded_count:
        issue("building_count_mismatch", "Source buildings do not equal encoded plus excluded")
    exclusions = record.get("exclusions", {})
    excluded_objects = exclusions.get("buildings", [])
    excluded_id_list = [str(item.get("building_id")) for item in excluded_objects]
    excluded_ids = set(excluded_id_list)
    if len(excluded_id_list) != len(excluded_ids):
        issue("duplicate_exclusion_id", "Excluded building IDs are not unique")
    if len(excluded_objects) != excluded_count:
        issue("excluded_object_count_mismatch", "Excluded object count differs from graph.excluded_building_count")
    exclusion_summary = exclusions.get("summary", {})
    actual_reason_counts = dict(sorted(Counter(str(item.get("reason")) for item in excluded_objects).items()))
    if exclusion_summary.get("excluded_building_reason_counts") != actual_reason_counts:
        issue("exclusion_reason_count_mismatch", "Excluded building reason counts differ from the object ledger")
    graph_counts = {
        "source_building_count": source_count,
        "encoded_building_count": encoded_count,
        "excluded_building_count": excluded_count,
    }
    if any(exclusion_summary.get(key) != value for key, value in graph_counts.items()):
        issue("exclusion_summary_mismatch", "Exclusion summary building counts differ from graph counts")
    leaked = excluded_ids & (set(node_ids) | set(cell_ids) | set(failure_ids) | {str(e.get("source")) for e in edges} | {str(e.get("target")) for e in edges})
    if leaked:
        issue("excluded_building_leak", f"Excluded buildings remain in Graph objects: {sorted(leaked)}")
    weak = record.get("weak_edge_summary", {})
    weak_fields = (
        ("long_distance_candidate_count", "long_distance_candidate"),
        ("low_shared_candidate_count", "low_shared_candidate"),
        ("combined_weak_candidate_count", "combined_weak_candidate"),
    )
    for count_field, flag_field in weak_fields:
        actual = sum(bool(edge.get(flag_field)) for edge in edges)
        if weak.get(count_field) != actual:
            issue("weak_edge_count_mismatch", f"{count_field} differs from edge flags")
    paths = _nonfinite_paths(record)
    if paths:
        issue("nonfinite_numeric", f"Non-finite numeric values at {paths[:5]}")

    components, isolated, average_degree = graph_connectivity(node_ids, edges)
    node_count, edge_count = len(nodes), len(edges)
    density = 0.0 if node_count < 2 else 2.0 * edge_count / (node_count * (node_count - 1))
    def ratio(numerator: Any, denominator: Any) -> float | None:
        return None if not denominator else float(numerator or 0) / float(denominator)
    values = record.get("block_attributes", {}).get("values", {})
    row = {
        **identity, "country": sample.get("country", graph.get("country", "")),
        "case_status": status_row.get("status"), "graph_path": status_row.get("graph_path", ""),
        "encoding_version": record.get("encoding_version"),
        **{key: actual_meta.get(key) for key in META_KEYS if key.endswith("sha256")},
        "graph_exists": True, "graph_readable": True, "file_bytes": int(float(status_row.get("output_bytes") or 0)),
        "validation_passed": record.get("validation", {}).get("passed") is True,
        "source_building_count": source_count, "encoded_building_count": encoded_count,
        "excluded_building_count": excluded_count, "node_count": node_count, "cell_count": len(cells),
        "cell_failure_count": len(failures), "edge_count": edge_count,
        "connected_components": components, "isolated_node_count": isolated,
        "average_degree": average_degree, "edge_density": density,
        "long_distance_candidate_count": weak.get("long_distance_candidate_count", 0),
        "low_shared_candidate_count": weak.get("low_shared_candidate_count", 0),
        "combined_weak_candidate_count": weak.get("combined_weak_candidate_count", 0),
        "long_distance_candidate_ratio": ratio(weak.get("long_distance_candidate_count", 0), edge_count) or 0.0,
        "low_shared_candidate_ratio": ratio(weak.get("low_shared_candidate_count", 0), edge_count) or 0.0,
        "combined_weak_candidate_ratio": ratio(weak.get("combined_weak_candidate_count", 0), edge_count) or 0.0,
        "excluded_building_ratio": ratio(excluded_count, source_count),
        "isolated_node_ratio": ratio(isolated, node_count),
        **{field: values.get(field) for field in (
            "area_m2", "elongation", "compactness", "lod1_building_count", "bcr", "far", "avh_w", "hcv", "orientation_entropy"
        )},
    }
    row["audit_issue_count"] = len(issues)
    row["audit_passed"] = not issues
    return row, issues


class FeatureAccumulator:
    """Compact per-field counters; no source objects are retained."""

    def __init__(self, registry=None) -> None:
        self.registry = registry if registry is not None else load_registry(PHASE_DIR / "feature_registry_snapshot_v03.tsv")
        self.stats: dict[tuple[str, str], dict[str, Any]] = defaultdict(
            lambda: {"object_count": 0, "observed_count": 0, "missing_count": 0, "nonfinite_count": 0,
                     "numeric_count": 0, "numeric_sum": 0.0, "numeric_sum_squares": 0.0,
                     "numeric_min": None, "numeric_max": None}
        )
        self.issues: list[dict] = []

    @staticmethod
    def _graph_value(field: str, record: dict, index_row: dict) -> Any:
        sample, graph, weak = record.get("sample", {}), record.get("graph", {}), record.get("weak_edge_summary", {})
        name = field.split(".", 1)[1]
        values = {
            "sample_id": sample.get("sample_id"), "block_id": sample.get("block_id", graph.get("block_id")),
            "city": sample.get("city", graph.get("city")), "country": sample.get("country", graph.get("country")),
            "sample_groups": "full_v032_population", "storage_srid": graph.get("storage_srid"),
            "metric_srid": graph.get("metric_srid"), "block_geometry_wkt_metric": graph.get("block_geometry_wkt_metric"),
            "source_block_relation": graph.get("source_index", {}).get("block_relation"),
            "source_building_relation": graph.get("source_index", {}).get("building_relation"),
            "quality_flags": graph.get("quality_flags"),
        }
        if name in values:
            return values[name]
        if name in index_row:
            return index_row[name]
        return weak.get(name)

    def update(self, record: dict, index_row: dict) -> None:
        city = str(index_row.get("city", ""))
        for registry_row in self.registry.to_dict("records"):
            entity, field = str(registry_row["entity"]), str(registry_row["field"])
            objects = [record] if entity in {"graph", "block"} else list(record.get(f"{entity}s", []))
            for obj in objects:
                if entity == "graph":
                    value, masked = self._graph_value(field, record, index_row), False
                elif entity == "block":
                    name = field.split(".", 1)[1]
                    block = record.get("block_attributes", {})
                    value, masked = block.get("values", {}).get(name), bool(block.get("missing", {}).get(name))
                else:
                    name = field.split(".", 1)[1]
                    value = obj.get(name)
                    masked = bool(obj.get("missing", {}).get(name)) if name not in {"missing", "missing_reasons", "quality"} else value is None
                missing = value is None or masked
                for scope in ("__global__", city):
                    stat = self.stats[(scope, field)]
                    stat["object_count"] += 1
                    if missing:
                        stat["missing_count"] += 1
                    else:
                        stat["observed_count"] += 1
                        if isinstance(value, (int, float)) and not isinstance(value, bool):
                            if not math.isfinite(float(value)):
                                stat["nonfinite_count"] += 1
                            else:
                                number = float(value)
                                stat["numeric_count"] += 1
                                stat["numeric_sum"] += number
                                stat["numeric_sum_squares"] += number * number
                                stat["numeric_min"] = number if stat["numeric_min"] is None else min(stat["numeric_min"], number)
                                stat["numeric_max"] = number if stat["numeric_max"] is None else max(stat["numeric_max"], number)
                if missing and str(registry_row["missing_rule"]).strip().lower() == "not allowed":
                    self.issues.append({
                        "sample_id": index_row.get("sample_id", ""), "city": city,
                        "block_id": index_row.get("block_id", ""), "part_id": index_row.get("part_id", ""),
                        "issue_type": "required_feature_missing", "message": f"{field} is missing",
                    })

    def to_rows(self, scope: dict | str | None = None) -> list[dict]:
        scope_name = scope if isinstance(scope, str) else (scope or {}).get("city", "__global__")
        registry = {row["field"]: row for row in self.registry.to_dict("records")}
        rows: list[dict] = []
        for field in self.registry["field"]:
            stat = dict(self.stats[(str(scope_name), str(field))])
            total = stat["object_count"]
            rows.append({"scope": scope_name, **registry[field], **stat,
                         "missing_rate": stat["missing_count"] / total if total else None})
        return rows
