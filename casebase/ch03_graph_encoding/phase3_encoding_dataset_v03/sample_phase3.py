"""Deterministic sampling for Chapter 3 phase 3.

The pure selection function is independent of PostgreSQL so its behavior can
be tested with small fixtures. Database extraction and file writing are added
by the phase runner.
"""

from __future__ import annotations

import hashlib
import random
import csv
import gzip
import io
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from psycopg2 import sql


PHASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PHASE_DIR / "results"
CH03_DIR = PHASE_DIR.parent
sys.path.insert(0, str(CH03_DIR))

from shared.db_readonly import readonly_connection


METRIC_FIELDS = [
    "area_m2", "elongation", "compactness", "lod1_building_count", "bcr",
    "convexity", "rectangularity", "far", "bcd", "osr", "avh", "avh_w",
    "hsd", "hcv", "h_max", "h_range", "total_volume", "avg_volume",
    "avg_footprint", "footprint_cv", "avg_complexity", "avg_rectangularity",
    "avg_convexity", "avg_orientation", "orientation_entropy", "avg_nn_distance",
]


@dataclass(frozen=True)
class Phase3SamplingConfig:
    seed: int = 20260920
    core_per_city: int = 40
    core_min_buildings: int = 2
    core_max_buildings: int = 499
    strata_per_city: int = 4
    single_stress_per_city: int = 1
    large_target_buildings: int = 500

    @property
    def per_stratum(self) -> int:
        if self.core_per_city % self.strata_per_city:
            raise ValueError("core_per_city must be divisible by strata_per_city")
        return self.core_per_city // self.strata_per_city


def _stable_seed(seed: int, *parts: Any) -> int:
    material = "|".join([str(seed), *(str(part) for part in parts)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def _key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row["city"]), str(row["block_id"])


def _ordered_sample(
    rows: Iterable[dict[str, Any]], count: int, seed: int, *seed_parts: Any
) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=_key)
    if len(ordered) <= count:
        return ordered
    rng = random.Random(_stable_seed(seed, *seed_parts))
    return sorted(rng.sample(ordered, count), key=_key)


def _select_from_stratum_pools(
    pools: dict[int, list[dict[str, Any]]], city: str, config: Phase3SamplingConfig
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Apply quotas and deterministic adjacent-stratum supplementation."""
    selections: list[dict[str, Any]] = []
    log: list[dict[str, Any]] = []
    used: set[tuple[str, str]] = set()
    deficits: list[tuple[int, int]] = []
    for target in range(1, config.strata_per_city + 1):
        initial = _ordered_sample(
            pools[target], config.per_stratum, config.seed, city, "core", target
        )
        for row in initial:
            used.add(_key(row))
            selections.append(
                {
                    "row": row,
                    "assigned_stratum": target,
                    "selection_stratum": target,
                    "supplemented": False,
                    "reason": f"core_stratum_{target}",
                }
            )
        deficits.append((target, config.per_stratum - len(initial)))

    for target, deficit in deficits:
        supplied = 0
        if deficit:
            neighbor_order = sorted(
                (index for index in pools if index != target),
                key=lambda index: (abs(index - target), index),
            )
            for neighbor in neighbor_order:
                available = [row for row in pools[neighbor] if _key(row) not in used]
                chosen = _ordered_sample(
                    available,
                    deficit - supplied,
                    config.seed,
                    city,
                    "supplement",
                    target,
                    neighbor,
                )
                for row in chosen:
                    used.add(_key(row))
                    selections.append(
                        {
                            "row": row,
                            "assigned_stratum": neighbor,
                            "selection_stratum": target,
                            "supplemented": True,
                            "reason": f"core_stratum_{target}_supplement_from_{neighbor}",
                        }
                    )
                supplied += len(chosen)
                if supplied == deficit:
                    break
            log.append(
                {
                    "city": city,
                    "target_stratum": target,
                    "deficit": deficit,
                    "supplied": supplied,
                    "unresolved": deficit - supplied,
                }
            )
    return selections, log


def select_phase3_samples(
    candidate_rows: Iterable[dict[str, Any]],
    regression_anchors: Iterable[dict[str, Any]] | None = None,
    config: Phase3SamplingConfig | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Select core, stress, and regression samples without duplicate cases."""
    config = config or Phase3SamplingConfig()
    rows = [dict(row) for row in candidate_rows]
    rows.sort(key=_key)
    indexed = {_key(row): row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError("candidate rows must be unique by city and block_id")

    frame: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        item = dict(row)
        building_count = int(row["lod1_building_count"])
        if building_count == 1:
            initial_reason = "not_selected_single_stress_quota"
        elif building_count >= config.large_target_buildings:
            initial_reason = "not_selected_large_stress_rule"
        else:
            initial_reason = "not_eligible"
        item.update(
            {
                "selected": False,
                "selection_reason": initial_reason,
                "sample_groups": [],
                "regression_anchor_ids": [],
                "assigned_stratum": None,
                "selection_stratum": None,
                "supplemented": False,
            }
        )
        frame[_key(row)] = item

    selected: dict[tuple[str, str], dict[str, Any]] = {}
    supplementation_log: list[dict[str, Any]] = []

    def add_group(
        row: dict[str, Any],
        group: str,
        reason: str,
        *,
        assigned_stratum: int | None = None,
        selection_stratum: int | None = None,
        supplemented: bool = False,
        anchor_id: str | None = None,
    ) -> None:
        key = _key(row)
        item = selected.setdefault(key, frame[key])
        item["selected"] = True
        if group not in item["sample_groups"]:
            item["sample_groups"].append(group)
        reasons = [] if item["selection_reason"].startswith("not_") else item["selection_reason"].split(";")
        if reason not in reasons:
            reasons.append(reason)
        item["selection_reason"] = ";".join(reasons)
        if assigned_stratum is not None:
            item["assigned_stratum"] = assigned_stratum
        if selection_stratum is not None:
            item["selection_stratum"] = selection_stratum
        item["supplemented"] = bool(item["supplemented"] or supplemented)
        if anchor_id and anchor_id not in item["regression_anchor_ids"]:
            item["regression_anchor_ids"].append(anchor_id)

    cities = sorted({str(row["city"]) for row in rows})
    strata_summary: list[dict[str, Any]] = []
    for city in cities:
        eligible = [
            row
            for row in rows
            if row["city"] == city
            and config.core_min_buildings <= int(row["lod1_building_count"]) <= config.core_max_buildings
        ]
        if eligible:
            pools: dict[int, list[dict[str, Any]]] = {index: [] for index in range(1, config.strata_per_city + 1)}
            ordered_eligible = sorted(
                eligible,
                key=lambda row: (int(row["lod1_building_count"]), str(row["block_id"])),
            )
            for position, row in enumerate(ordered_eligible):
                assigned = min(
                    position * config.strata_per_city // len(ordered_eligible) + 1,
                    config.strata_per_city,
                )
                frame[_key(row)]["assigned_stratum"] = assigned
                frame[_key(row)]["selection_reason"] = "not_selected_core_quota"
                pools[assigned].append(row)
            core_selections, city_log = _select_from_stratum_pools(pools, city, config)
            supplementation_log.extend(city_log)
            for choice in core_selections:
                add_group(
                    choice["row"],
                    "core",
                    choice["reason"],
                    assigned_stratum=choice["assigned_stratum"],
                    selection_stratum=choice["selection_stratum"],
                    supplemented=choice["supplemented"],
                )

            for stratum, pool in pools.items():
                strata_summary.append(
                    {
                        "city": city,
                        "stratum": stratum,
                        "candidate_count": len(pool),
                        "selected_for_stratum": sum(
                            item["city"] == city
                            and item["selection_stratum"] == stratum
                            and "core" in item["sample_groups"]
                            for item in selected.values()
                        ),
                        "min_building_count": min(
                            (int(row["lod1_building_count"]) for row in pool), default=None
                        ),
                        "max_building_count": max(
                            (int(row["lod1_building_count"]) for row in pool), default=None
                        ),
                    }
                )

        singles = [row for row in rows if row["city"] == city and int(row["lod1_building_count"]) == 1]
        for row in _ordered_sample(singles, config.single_stress_per_city, config.seed, city, "single"):
            add_group(row, "stress_single", "stress_single_deterministic")

        large = [row for row in rows if row["city"] == city and int(row["lod1_building_count"]) >= 500]
        if large:
            row = min(
                large,
                key=lambda item: (
                    abs(int(item["lod1_building_count"]) - config.large_target_buildings),
                    int(item["lod1_building_count"]),
                    str(item["block_id"]),
                ),
            )
            add_group(row, "stress_large", "stress_large_closest_to_500")

    missing_anchors: list[dict[str, Any]] = []
    for anchor in regression_anchors or []:
        key = str(anchor["city"]), str(anchor["block_id"])
        if key not in indexed:
            missing_anchors.append(dict(anchor))
            continue
        add_group(
            indexed[key],
            "regression_anchor",
            "phase2_regression_anchor",
            anchor_id=str(anchor["sample_id"]),
        )

    return {
        "selected": sorted(selected.values(), key=_key),
        "frame": [frame[_key(row)] for row in rows],
        "strata_summary": strata_summary,
        "supplementation_log": supplementation_log,
        "missing_anchors": missing_anchors,
    }


def _csv_value(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return ";".join(str(item) for item in value)
    if isinstance(value, bool):
        return int(value)
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".gz":
        raw = path.open("wb")
        compressed = gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0)
        handle = io.TextIOWrapper(compressed, encoding="utf-8-sig", newline="")
    else:
        handle = path.open(mode="w", encoding="utf-8-sig", newline="")
    with handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: _csv_value(row.get(name)) for name in fieldnames})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _quality_fields(row: dict[str, Any]) -> tuple[list[str], list[str]]:
    missing = [field for field in METRIC_FIELDS if row.get(field) is None]
    flags: list[str] = []
    if row.get("area_m2") is None or float(row["area_m2"]) <= 0:
        flags.append("nonpositive_or_missing_block_area")
    if row.get("bcr") is not None and float(row["bcr"]) > 1:
        flags.append("bcr_gt_1")
    if row.get("avg_nn_distance") is not None and float(row["avg_nn_distance"]) == 0:
        flags.append("avg_nn_distance_zero")
    if missing:
        flags.append("candidate_metrics_missing")
    return flags, missing


def fetch_candidate_rows(connection) -> list[dict[str, Any]]:
    selected_columns = sql.SQL(", ").join(
        sql.SQL("v.{} AS {}").format(sql.Identifier(name), sql.Identifier(name))
        for name in METRIC_FIELDS
    )
    statement = sql.SQL(
        """
        SELECT v.block_id, v.city, v.country, m.city_query,
               m.lod1_metric_srid AS metric_srid, ST_SRID(v.geom) AS storage_srid,
               {selected_columns}
        FROM summary.lod1_candidate_blocks v
        JOIN summary.city_name_map m ON m.city = v.city
        ORDER BY v.city, v.block_id
        """
    ).format(selected_columns=selected_columns)
    with connection.cursor() as cursor:
        cursor.execute(statement)
        columns = [item.name for item in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]


def read_regression_anchors() -> list[dict[str, str]]:
    path = PHASE_DIR.parent / "phase2_5_tessellation_relations_v02" / "results" / "input_manifest.csv"
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return [
            {"sample_id": row["sample_id"], "city": row["city"], "block_id": row["block_id"]}
            for row in csv.DictReader(handle)
        ]


def _verify_building_counts(connection, selected: list[dict[str, Any]]) -> None:
    cities = sorted({str(row["city"]) for row in selected})
    for city in cities:
        city_rows = [row for row in selected if row["city"] == city]
        relation = str(city_rows[0]["city_query"]) + "_buildings_lod1"
        ids = [str(row["block_id"]) for row in city_rows]
        statement = sql.SQL(
            "SELECT block_id, COUNT(*) AS actual_count FROM lod1.{} WHERE block_id = ANY(%s) GROUP BY block_id"
        ).format(sql.Identifier(relation))
        with connection.cursor() as cursor:
            cursor.execute(statement, (ids,))
            counts = {str(block_id): int(count) for block_id, count in cursor.fetchall()}
        for row in city_rows:
            row["verified_building_record_count"] = counts.get(str(row["block_id"]), 0)
            row["building_count_matches"] = (
                row["verified_building_record_count"] == int(row["lod1_building_count"])
            )


def _manifest_rows(selected: list[dict[str, Any]]) -> list[dict[str, Any]]:
    group_names = {
        "core": "core_audit",
        "stress_single": "single_building_stress",
        "stress_large": "large_graph_stress",
        "regression_anchor": "regression_anchor",
    }
    rows = []
    for item in selected:
        row = dict(item)
        flags, missing = _quality_fields(row)
        row.update(
            {
                "sample_id": f"p3__{row['block_id']}",
                "sample_groups": [group_names[name] for name in row["sample_groups"]],
                "quality_flags": flags,
                "missing_metric_fields": missing,
                "source_block_relation": "summary.lod1_candidate_blocks",
                "source_building_relation": f"lod1.{row['city_query']}_buildings_lod1",
                "dataset_role": "development_audit",
            }
        )
        rows.append(row)
    return rows


def _frame_rows(frame: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for item in frame:
        row = dict(item)
        flags, missing = _quality_fields(row)
        row.update(
            {
                "quality_flags": flags,
                "missing_metric_fields": missing,
                "source_block_relation": "summary.lod1_candidate_blocks",
                "source_building_relation": f"lod1.{row['city_query']}_buildings_lod1",
            }
        )
        rows.append(row)
    return rows


def _database_snapshot(connection, candidates: list[dict[str, Any]]) -> dict[str, Any]:
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT current_database(), version(), current_setting('transaction_read_only'), PostGIS_Full_Version()"
        )
        database, version, read_only, postgis = cursor.fetchone()
    per_city = []
    for city in sorted({str(row["city"]) for row in candidates}):
        city_rows = [row for row in candidates if row["city"] == city]
        counts = [int(row["lod1_building_count"]) for row in city_rows]
        per_city.append(
            {
                "city": city,
                "candidate_count": len(city_rows),
                "min_lod1_building_count": min(counts),
                "max_lod1_building_count": max(counts),
            }
        )
    return {
        "experiment_id": "CH3-P3-ENCODING-SCALE-V03",
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "database": database,
        "postgresql_version": version,
        "postgis_version": postgis,
        "transaction_read_only": read_only,
        "candidate_relation": "summary.lod1_candidate_blocks",
        "candidate_query_order": ["city", "block_id"],
        "candidate_record_count": len(candidates),
        "city_count": len(per_city),
        "metric_fields": METRIC_FIELDS,
        "per_city": per_city,
    }


def main() -> int:
    config = Phase3SamplingConfig()
    with readonly_connection(statement_timeout_ms=600_000) as connection:
        candidates = fetch_candidate_rows(connection)
        result = select_phase3_samples(candidates, read_regression_anchors(), config)
        if result["missing_anchors"]:
            raise RuntimeError(f"Regression anchors missing: {result['missing_anchors']}")
        manifest = _manifest_rows(result["selected"])
        _verify_building_counts(connection, manifest)
        snapshot = _database_snapshot(connection, candidates)

    manifest_fields = [
        "sample_id", "city", "country", "block_id", "lod1_building_count",
        "sample_groups", "regression_anchor_ids", "selection_reason",
        "assigned_stratum", "selection_stratum", "supplemented",
        "dataset_role", "quality_flags", "missing_metric_fields",
        "source_block_relation", "source_building_relation", "storage_srid",
        "metric_srid", "verified_building_record_count", "building_count_matches",
        *[name for name in METRIC_FIELDS if name != "lod1_building_count"],
    ]
    manifest_path = RESULTS_DIR / "sample_manifest.csv"
    _write_csv(manifest_path, manifest, manifest_fields)

    frame = _frame_rows(result["frame"])
    frame_fields = [
        "city", "country", "block_id", "lod1_building_count", "selected",
        "selection_reason", "sample_groups", "assigned_stratum", "selection_stratum",
        "supplemented", "quality_flags", "missing_metric_fields",
        "source_block_relation", "source_building_relation", "storage_srid", "metric_srid",
        *[name for name in METRIC_FIELDS if name != "lod1_building_count"],
    ]
    frame_path = RESULTS_DIR / "sampling_frame.csv.gz"
    _write_csv(frame_path, frame, frame_fields)

    selected_keys = {_key(row) for row in manifest}
    summary_rows = []
    for city in sorted({str(row["city"]) for row in candidates}):
        city_frame = [row for row in frame if row["city"] == city]
        city_manifest = [row for row in manifest if row["city"] == city]
        summary_rows.append(
            {
                "city": city,
                "candidate_count": len(city_frame),
                "core_eligible_count": sum(config.core_min_buildings <= int(row["lod1_building_count"]) <= config.core_max_buildings for row in city_frame),
                "selected_unique_count": len(city_manifest),
                "core_count": sum("core_audit" in row["sample_groups"] for row in city_manifest),
                "single_stress_count": sum("single_building_stress" in row["sample_groups"] for row in city_manifest),
                "large_stress_count": sum("large_graph_stress" in row["sample_groups"] for row in city_manifest),
                "regression_anchor_count": sum("regression_anchor" in row["sample_groups"] for row in city_manifest),
                "unselected_count": sum(_key(row) not in selected_keys for row in city_frame),
            }
        )
    _write_csv(
        RESULTS_DIR / "sampling_frame_summary.csv",
        summary_rows,
        list(summary_rows[0]),
    )
    _write_csv(
        RESULTS_DIR / "sampling_strata.csv",
        result["strata_summary"],
        ["city", "stratum", "candidate_count", "selected_for_stratum", "min_building_count", "max_building_count"],
    )
    _write_csv(
        RESULTS_DIR / "sampling_supplementation.csv",
        result["supplementation_log"],
        ["city", "target_stratum", "deficit", "supplied", "unresolved"],
    )

    snapshot.update(
        {
            "sampling_config": config.__dict__,
            "selected_unique_count": len(manifest),
            "sample_manifest_sha256": _sha256(manifest_path),
            "sampling_frame_sha256": _sha256(frame_path),
            "building_count_mismatch_count": sum(not row["building_count_matches"] for row in manifest),
        }
    )
    (RESULTS_DIR / "database_snapshot.json").write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "candidate_rows": len(candidates),
                "selected_unique": len(manifest),
                "manifest_sha256": snapshot["sample_manifest_sha256"],
                "building_count_mismatches": snapshot["building_count_mismatch_count"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
