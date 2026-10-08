"""Deterministic, within-city matching using only frozen index attributes."""

from __future__ import annotations

from collections import Counter
import hashlib

import numpy as np
import pandas as pd


IDENTITY_FIELDS = ("sample_id", "city", "country", "block_id", "part_id", "graph_path")
COUNT_FIELDS = ("lod1_building_count", "source_building_count", "excluded_building_count", "cell_failure_count")
PAIR_COLUMNS = (
    "pair_id", "city", "sample_id_a", "sample_id_b", "block_id_a", "block_id_b",
    "part_id_a", "part_id_b", "graph_path_a", "graph_path_b", "candidate_level",
    "mean_standardized_difference", "max_standardized_difference", "fields_used",
)


def parse_bool(value) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None or pd.isna(value):
        return False
    text = str(value).strip().lower()
    if text in ("true", "1", "1.0"):
        return True
    if text in ("false", "0", "0.0", ""):
        return False
    raise ValueError(f"Unsupported boolean value: {value!r}")


def sample_development(index: pd.DataFrame, spec: dict) -> pd.DataFrame:
    required = {"sample_id", "city", "case_status", "audit_passed", *spec["pair_fields"]}
    if required - set(index):
        raise ValueError(f"Missing index fields: {sorted(required - set(index))}")
    if index["sample_id"].astype(str).duplicated().any():
        raise ValueError("The source index contains duplicate sample IDs")
    flags = index["audit_passed"].map(parse_bool)
    eligible = index["case_status"].eq("completed") & flags
    columns = [field for field in (*IDENTITY_FIELDS, *spec["pair_fields"], *COUNT_FIELDS) if field in index]
    frame = index.loc[eligible, columns].copy()
    if frame["sample_id"].isna().any() or frame["city"].isna().any():
        raise ValueError("Missing identity in the source index")
    frame["sample_id"] = frame["sample_id"].astype(str)
    frame["city"] = frame["city"].astype(str)
    frame["sampling_sha256"] = frame["sample_id"].map(
        lambda value: hashlib.sha256((spec["seed_material"] + value).encode("utf-8")).hexdigest()
    )
    cap = int(spec["max_per_city"])
    if not 1 <= cap <= 1000:
        raise ValueError("Development sampling must use 1..1000 cases per city")
    frame = frame.sort_values(["city", "sampling_sha256", "sample_id"], kind="stable")
    frame = frame.groupby("city", sort=True).head(cap).reset_index(drop=True)
    frame.attrs["summary"] = {
        "source_entries": len(index), "eligible_entries": int(eligible.sum()),
        "source_status_counts": {str(key): int(value) for key, value in index["case_status"].value_counts().items()},
        "completed_audit_not_passed": int((index["case_status"].eq("completed") & ~flags).sum()),
        "development_entries": len(frame), "per_city": {str(key): int(value) for key, value in frame.groupby("city").size().items()},
        "algorithm": "SHA256(seed_material + sample_id), then sample_id; independently within city",
        "is_training_split": False,
    }
    return frame


def select_candidate_pairs(development: pd.DataFrame, spec: dict) -> pd.DataFrame:
    fields = list(spec["pair_fields"])
    pools = {level: {} for level in ("close", "relaxed", "nearest_only")}
    city_reports: dict = {}
    for city, group in development.groupby("city", sort=True):
        group = group.sort_values("sample_id", kind="stable").reset_index(drop=True)
        values = pd.DataFrame(index=group.index)
        for field, rule in spec["pair_fields"].items():
            column = pd.to_numeric(group[field], errors="coerce").replace([np.inf, -np.inf], np.nan)
            if rule["transform"] == "log":
                column = np.log(column.where(column > 0))
            values[field] = column
        medians = values.median()
        iqrs = values.quantile(0.75) - values.quantile(0.25)
        active = [field for field in fields if pd.notna(iqrs[field]) and iqrs[field] > 0]
        report = {
            "development_entries": len(group), "active_fields": active,
            "median": {field: float(medians[field]) if pd.notna(medians[field]) else None for field in fields},
            "iqr": {field: float(iqrs[field]) if pd.notna(iqrs[field]) else None for field in fields},
            "missing_required_fields": 0, "candidate_counts": {}, "reason": "",
        }
        city_reports[str(city)] = report
        if len(active) < int(spec["min_active_fields"]):
            report["reason"] = "insufficient_active_fields"
            continue
        valid = values[active].notna().all(axis=1)
        report["missing_required_fields"] = int((~valid).sum())
        group = group.loc[valid].reset_index(drop=True)
        if len(group) < 2:
            report["reason"] = "fewer_than_two_complete_cases"
            continue
        matrix = ((values.loc[valid, active] - medians[active]) / iqrs[active]).to_numpy(dtype=float)
        ids = group["sample_id"].astype(str).to_numpy()
        undirected: dict = {}
        # One city's cases only. No cross-city or full-library distance matrix.
        for i in range(len(group)):
            differences = np.abs(matrix - matrix[i])
            means = differences.mean(axis=1)
            means[i] = np.inf
            neighbors = np.lexsort((ids, means))[:min(int(spec["nearest_neighbors"]), len(group) - 1)]
            for j in neighbors:
                a, b = (i, j) if ids[i] < ids[j] else (j, i)
                key = (str(ids[a]), str(ids[b]))
                if key in undirected:
                    continue
                maximum = float(differences[j].max())
                level = "close" if maximum <= spec["close_max"] else (
                    "relaxed" if maximum <= spec["relaxed_max"] else "nearest_only"
                )
                row = {
                    "city": str(city), "candidate_level": level,
                    "mean_standardized_difference": float(means[j]), "max_standardized_difference": maximum,
                    "fields_used": ";".join(active),
                }
                for side, position in (("a", a), ("b", b)):
                    for name in (*IDENTITY_FIELDS, *fields, *COUNT_FIELDS):
                        if name in group:
                            value = group.iloc[position][name]
                            row[f"{name}_{side}"] = None if pd.isna(value) else value
                for field in fields:
                    row[f"std_diff_{field}"] = float(abs(matrix[a, active.index(field)] - matrix[b, active.index(field)])) if field in active else None
                undirected[key] = row
        report["candidate_counts"] = dict(Counter(row["candidate_level"] for row in undirected.values()))
        for level in pools:
            pools[level][str(city)] = sorted(
                (row for row in undirected.values() if row["candidate_level"] == level),
                key=lambda row: (row["mean_standardized_difference"], row["sample_id_a"], row["sample_id_b"]),
            )
    chosen: list[dict] = []
    used: set[str] = set()
    city_counts: Counter = Counter()
    target = min(int(spec["max_pairs"]), 30)
    def take_level(level: str, limit: int) -> None:
        queues = {city: iter(rows) for city, rows in sorted(pools[level].items())}
        while len(chosen) < limit:
            added = False
            for city, queue in queues.items():
                if city_counts[city] >= min(int(spec["max_pairs_per_city"]), 3):
                    continue
                for row in queue:
                    ids = {row["sample_id_a"], row["sample_id_b"]}
                    if not ids & used:
                        chosen.append(row); used.update(ids); city_counts[city] += 1
                        added = True
                        break
                if len(chosen) >= limit:
                    return
            if not added:
                return
    take_level("close", target)
    take_level("relaxed", target)
    fallback_target = min(int(spec["nearest_only_min_pairs"]), target)
    if len(chosen) < fallback_target:
        take_level("nearest_only", fallback_target)
    for i, row in enumerate(chosen, 1):
        row["pair_id"] = f"P6A_{i:03d}"
    frame = pd.DataFrame(chosen) if chosen else pd.DataFrame(columns=PAIR_COLUMNS)
    if not frame.empty:
        frame = frame[[*PAIR_COLUMNS, *[name for name in frame if name not in PAIR_COLUMNS]]]
    frame.attrs["summary"] = {
        "cities": city_reports, "selected_pairs": len(chosen), "target_pairs": target,
        "selected_by_city": dict(city_counts), "selected_by_level": dict(Counter(row["candidate_level"] for row in chosen)),
        "shortfall": max(0, target - len(chosen)),
        "shortfall_policy": "Deliver actual count; no imputation, extra sampling or relaxed thresholds",
    }
    return frame
