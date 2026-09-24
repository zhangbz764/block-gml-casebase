"""Merge Phase 5 part shards into the frozen all-entry index and compact summaries."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd


QUANTILES = [("min", 0.0), ("p01", 0.01), ("p05", 0.05), ("p25", 0.25), ("p50", 0.5),
             ("p75", 0.75), ("p95", 0.95), ("p99", 0.99), ("max", 1.0)]
STRUCTURE_FIELDS = [
    "node_count", "edge_count", "average_degree", "edge_density", "connected_components",
    "isolated_node_ratio", "long_distance_candidate_ratio", "low_shared_candidate_ratio",
    "combined_weak_candidate_ratio", "excluded_building_ratio",
]
SUM_FIELDS = ["object_count", "observed_count", "missing_count", "nonfinite_count", "numeric_count", "numeric_sum", "numeric_sum_squares"]


def merge_part_indexes(part_paths: list[Path], expected_rows: int) -> pd.DataFrame:
    missing = [str(path) for path in part_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing part index shards: {missing}")
    index = pd.concat([pd.read_csv(path, encoding="utf-8-sig", low_memory=False) for path in part_paths], ignore_index=True)
    if len(index) != int(expected_rows):
        raise ValueError(f"Merged index has {len(index)} rows, expected {expected_rows}")
    if index["sample_id"].astype(str).duplicated().any():
        raise ValueError("Merged index has duplicate sample_id")
    if index[["city", "block_id"]].astype(str).duplicated().any():
        raise ValueError("Merged index has duplicate city + block_id")
    return index.sort_values(["part_id", "city", "block_id", "sample_id"], kind="mergesort").reset_index(drop=True)


def merge_feature_counts(paths: list[Path]) -> pd.DataFrame:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing feature count shards: {missing}")
    frame = pd.concat([pd.read_csv(path, encoding="utf-8-sig") for path in paths], ignore_index=True)
    identity = ["scope", "entity", "field", "role", "unit", "transform_or_encoding", "missing_rule", "default_input", "rationale"]
    for field in SUM_FIELDS:
        frame[field] = pd.to_numeric(frame[field], errors="coerce").fillna(0)
    aggregate = {field: "sum" for field in SUM_FIELDS}
    aggregate.update({"numeric_min": "min", "numeric_max": "max"})
    merged = frame.groupby(identity, dropna=False, as_index=False).agg(aggregate)
    merged["missing_rate"] = merged["missing_count"].div(merged["object_count"].where(merged["object_count"].ne(0)))
    return merged.sort_values(["scope", "entity", "field"], kind="mergesort").reset_index(drop=True)


def _quantile_rows(frame: pd.DataFrame, scope_name: str, scope_value: str) -> list[dict]:
    rows: list[dict] = []
    for field in STRUCTURE_FIELDS:
        values = pd.to_numeric(frame[field], errors="coerce").dropna()
        for label, quantile in QUANTILES:
            rows.append({"scope": scope_name, "scope_value": scope_value, "field": field,
                         "quantile": label, "value": values.quantile(quantile) if len(values) else None,
                         "graph_count": len(values)})
    return rows


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig", lineterminator="\n")
    os.replace(temporary, path)


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_dataset_summaries(index: pd.DataFrame, feature_counts: pd.DataFrame, output_dir: Path) -> dict:
    completed = index.loc[index["case_status"].eq("completed")].copy()
    excluded = index.loc[index["case_status"].eq("excluded_no_valid_building")]
    global_summary = {
        "entry_count": int(len(index)), "city_count": int(index["city"].nunique()),
        "completed_graph_count": int(len(completed)), "excluded_no_valid_building_count": int(len(excluded)),
        "hard_issue_entry_count": int((pd.to_numeric(index["audit_issue_count"], errors="coerce").fillna(0) > 0).sum()),
        "node_total": int(pd.to_numeric(completed["node_count"], errors="coerce").sum()),
        "edge_total": int(pd.to_numeric(completed["edge_count"], errors="coerce").sum()),
        "cell_failure_total": int(pd.to_numeric(completed["cell_failure_count"], errors="coerce").sum()),
        "excluded_building_total": int(pd.to_numeric(index["excluded_building_count"], errors="coerce").sum()),
        "status_counts": {str(key): int(value) for key, value in index.groupby("case_status").size().to_dict().items()},
    }
    city_rows: list[dict] = []
    city_quantiles: list[dict] = []
    for city, entries in index.groupby("city", sort=True):
        graphs = entries.loc[entries["case_status"].eq("completed")]
        row = {
            "city": city, "entry_count": len(entries), "completed_graph_count": len(graphs),
            "excluded_no_valid_building_count": int(entries["case_status"].eq("excluded_no_valid_building").sum()),
            "node_total": int(pd.to_numeric(graphs["node_count"], errors="coerce").sum()),
            "edge_total": int(pd.to_numeric(graphs["edge_count"], errors="coerce").sum()),
            "excluded_building_total": int(pd.to_numeric(entries["excluded_building_count"], errors="coerce").sum()),
        }
        for field in STRUCTURE_FIELDS:
            values = pd.to_numeric(graphs[field], errors="coerce").dropna()
            row[f"{field}_median"] = values.quantile(0.5) if len(values) else None
            row[f"{field}_p99"] = values.quantile(0.99) if len(values) else None
        city_rows.append(row)
        city_quantiles.extend(_quantile_rows(graphs, "city", str(city)))

    global_features = feature_counts.loc[feature_counts["scope"].eq("__global__")].drop(columns="scope")
    city_features = feature_counts.loc[~feature_counts["scope"].eq("__global__")].rename(columns={"scope": "city"})
    issue_files = sorted((output_dir / "parts").glob("part_*_issues.csv"))
    if issue_files:
        issues = pd.concat([pd.read_csv(path, encoding="utf-8-sig") for path in issue_files], ignore_index=True)
        issue_summary = issues.groupby(["issue_type"], as_index=False).size().rename(columns={"size": "issue_count"}) if not issues.empty else pd.DataFrame(columns=["issue_type", "issue_count"])
    else:
        issue_summary = pd.DataFrame(columns=["issue_type", "issue_count"])
    _atomic_json(output_dir / "global_summary.json", global_summary)
    _atomic_csv(output_dir / "summary_by_city.csv", pd.DataFrame(city_rows))
    _atomic_csv(output_dir / "structure_quantiles_global.csv", pd.DataFrame(_quantile_rows(completed, "global", "all")))
    _atomic_csv(output_dir / "structure_quantiles_by_city.csv", pd.DataFrame(city_quantiles))
    _atomic_csv(output_dir / "feature_completeness_global.csv", global_features)
    _atomic_csv(output_dir / "feature_completeness_by_city.csv", city_features)
    _atomic_csv(output_dir / "issue_summary.csv", issue_summary)
    return global_summary
