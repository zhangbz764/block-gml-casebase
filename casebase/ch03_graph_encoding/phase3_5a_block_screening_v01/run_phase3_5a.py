"""Database runner for the Phase 3.5A pilot and researcher-triggered full run."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.metadata
import json
import platform
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


PHASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PHASE_DIR / "results"
CH03_DIR = PHASE_DIR.parent
sys.path.insert(0, str(CH03_DIR))

from shared.db_readonly import readonly_connection
from screen_blocks import (
    EXPECTED_SINGLE_NULLS,
    METRIC_FIELDS,
    RULES,
    build_filter_summaries,
    screen_dataframe,
)


RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def validate_run_id(run_id: str) -> str:
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError("run_id must use only letters, digits, dot, underscore or hyphen")
    return run_id


def run_output_paths(mode: str, run_id: str) -> list[Path]:
    validate_run_id(run_id)
    audit_dir = RESULTS_DIR / "audit" / run_id
    if mode == "pilot":
        return [
            RESULTS_DIR / "pilot" / "pilot_manifest.csv",
            RESULTS_DIR / "pilot" / "pilot_screening_ledger.csv",
            audit_dir,
        ]
    return [
        RESULTS_DIR / "bulk" / run_id,
        audit_dir,
        RESULTS_DIR / "visual_review" / run_id,
    ]


def _unexpected_missing_mask(frame: pd.DataFrame) -> pd.Series:
    numeric = frame.loc[:, METRIC_FIELDS].apply(pd.to_numeric, errors="coerce")
    missing = numeric.isna()
    required_for_single = [field for field in METRIC_FIELDS if field not in EXPECTED_SINGLE_NULLS]
    return (
        numeric["lod1_building_count"].eq(1) & missing.loc[:, required_for_single].any(axis=1)
    ) | (
        numeric["lod1_building_count"].gt(1) & missing.any(axis=1)
    )


def select_pilot_frame(frame: pd.DataFrame, per_city: int = 10) -> pd.DataFrame:
    """Select the deterministic per-city baseline plus every known hard-anomaly class."""
    if per_city < 1:
        raise ValueError("per_city must be positive")
    ordered = frame.copy().sort_values(["city", "block_id"], kind="mergesort").reset_index(drop=True)
    baseline = ordered.groupby("city", sort=True).cumcount().lt(per_city)
    numeric = ordered.loc[:, METRIC_FIELDS].apply(pd.to_numeric, errors="coerce")
    coverage = numeric["bcr"].gt(1) | numeric["osr"].lt(0)
    duplicate_centroid = numeric["lod1_building_count"].gt(1) & numeric["avg_nn_distance"].eq(0)
    unexpected_missing = _unexpected_missing_mask(ordered)
    selected = baseline | coverage | duplicate_centroid | unexpected_missing
    result = ordered.loc[selected].copy()
    reason_map = {
        "city_head10": baseline,
        "coverage_inconsistent": coverage,
        "duplicate_centroid_signal": duplicate_centroid,
        "unexpected_metric_missing": unexpected_missing,
    }
    result["selection_reasons"] = [
        ";".join(name for name, mask in reason_map.items() if bool(mask.loc[index]))
        for index in result.index
    ]
    return result.sort_values(["city", "block_id"], kind="mergesort").reset_index(drop=True)


def stable_csv_bytes(frame: pd.DataFrame) -> bytes:
    text = frame.to_csv(index=False, lineterminator="\n", float_format="%.15g")
    return b"\xef\xbb\xbf" + text.encode("utf-8")


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(stable_csv_bytes(frame))


def _write_gzip_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            compressed.write(stable_csv_bytes(frame))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_database_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for field in METRIC_FIELDS:
        result[field] = pd.to_numeric(result[field], errors="coerce")
    for field in ("geom_is_null", "geom_is_empty", "geom_is_valid", "geom_area_positive"):
        result[field] = result[field].astype(bool)
    return result


def _fetch_candidate_frame(connection) -> pd.DataFrame:
    query = (PHASE_DIR / "query_candidate_blocks.sql").read_text(encoding="utf-8")
    with connection.cursor() as cursor:
        cursor.execute(query)
        columns = [item.name for item in cursor.description]
        chunks = []
        while True:
            rows = cursor.fetchmany(25_000)
            if not rows:
                break
            chunks.append(pd.DataFrame.from_records(rows, columns=columns))
    if not chunks:
        raise RuntimeError("Candidate query returned no rows")
    return _normalize_database_frame(pd.concat(chunks, ignore_index=True))


def _threshold_summary(thresholds: dict[str, float]) -> pd.DataFrame:
    mapping = (
        ("area_m2", "upper", "area_m2_p99"),
        ("elongation", "lower", "elongation_p01"),
        ("compactness", "lower", "compactness_p01"),
        ("lod1_building_count", "upper", "lod1_building_count_p99"),
        ("bcr", "lower", "bcr_p01"),
    )
    return pd.DataFrame(
        [
            {
                "metric": metric,
                "quantile": RULES["scope_quantiles"][side],
                "threshold": thresholds[key],
                "eligible_count": thresholds["eligible_count"],
                "rule_version": RULES["rule_version"],
            }
            for metric, side, key in mapping
        ]
    )


def _write_audit(run_id: str, ledger: pd.DataFrame) -> Path:
    audit_dir = RESULTS_DIR / "audit" / run_id
    summaries = build_filter_summaries(ledger)
    for name, frame in summaries.items():
        _write_csv(audit_dir / f"{name}.csv", frame)
    _write_csv(audit_dir / "threshold_summary.csv", _threshold_summary(ledger.attrs["thresholds"]))
    return audit_dir


def _source_hashes() -> dict[str, str]:
    names = (
        "screening_rules_v0.1.json",
        "query_candidate_blocks.sql",
        "screen_blocks.py",
        "run_phase3_5a.py",
        "verify_screening.py",
        "build_review_sample.py",
        "render_review_cases.py",
    )
    return {name: _sha256(PHASE_DIR / name) for name in names if (PHASE_DIR / name).exists()}


def _write_metadata(run_id: str, mode: str, database: dict[str, Any], source: pd.DataFrame, ledger_path: Path) -> None:
    now = datetime.now().astimezone().isoformat()
    environment = {
        "captured_at": now,
        "experiment_id": RULES["experiment_id"],
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "libraries": {
            name: importlib.metadata.version(name)
            for name in ("pandas", "numpy", "psycopg2", "matplotlib", "shapely", "geopandas")
        },
        "database": database,
    }
    (RESULTS_DIR / "environment.json").write_text(
        json.dumps(environment, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    query_path = PHASE_DIR / "query_candidate_blocks.sql"
    snapshot = {
        "captured_at": now,
        "run_id": run_id,
        "mode": mode,
        "source_relation": RULES["source_relation"],
        "transaction_read_only": database["transaction_read_only"],
        "candidate_record_count": int(len(source)),
        "city_count": int(source["city"].nunique(dropna=True)),
        "unique_city_block_count": int(source[["city", "block_id"]].drop_duplicates().shape[0]),
        "geometry_field": "geom",
        "metric_field_count": len(METRIC_FIELDS),
        "query_sha256": _sha256(query_path),
        "screening_rules_sha256": _sha256(PHASE_DIR / "screening_rules_v0.1.json"),
    }
    (RESULTS_DIR / "input_snapshot.json").write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    freeze = {
        "captured_at": now,
        "run_id": run_id,
        "mode": mode,
        "experiment_id": RULES["experiment_id"],
        "rule_version": RULES["rule_version"],
        "source_sha256": _source_hashes(),
        "ledger_path": str(ledger_path),
        "ledger_sha256": _sha256(ledger_path),
    }
    (RESULTS_DIR / "freeze.json").write_text(
        json.dumps(freeze, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def run(mode: str, run_id: str, expected_rows: int, expected_cities: int) -> dict[str, Any]:
    validate_run_id(run_id)
    audit_dir = RESULTS_DIR / "audit" / run_id
    output_paths = run_output_paths(mode, run_id)
    occupied = [path for path in output_paths if path.exists()]
    if occupied:
        raise FileExistsError(f"Refusing to overwrite existing output: {occupied}")

    log_path = RESULTS_DIR / "run_log.txt"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with readonly_connection(statement_timeout_ms=900_000) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SHOW transaction_read_only")
            read_only = cursor.fetchone()[0]
            cursor.execute("SELECT current_database(), version()")
            database_name, database_version = cursor.fetchone()
        if read_only != "on":
            raise RuntimeError("Database transaction is not read-only")
        source = _fetch_candidate_frame(connection)

    observed_rows = len(source)
    observed_cities = source["city"].nunique(dropna=True)
    unique_keys = source[["city", "block_id"]].drop_duplicates().shape[0]
    if observed_rows != expected_rows or observed_cities != expected_cities:
        raise RuntimeError(
            f"Frozen input mismatch: rows={observed_rows}/{expected_rows}, cities={observed_cities}/{expected_cities}"
        )
    if unique_keys != observed_rows:
        raise RuntimeError(f"city + block_id is not unique: {unique_keys}/{observed_rows}")

    if mode == "pilot":
        manifest = select_pilot_frame(source)
        ledger = screen_dataframe(manifest)
        h03_count = int(ledger["H03"].sum())
        h05_count = int(ledger["H05"].sum())
        if h03_count != 86 or h05_count != 2:
            raise RuntimeError(f"Known QA regression mismatch: H03={h03_count}/86, H05={h05_count}/2")
        if not ledger.loc[ledger["H03"] | ledger["H05"], "machine_status"].eq("exclude_hard").all():
            raise RuntimeError("Known pilot hard anomaly did not receive exclude_hard")
        manifest_path = RESULTS_DIR / "pilot" / "pilot_manifest.csv"
        ledger_path = RESULTS_DIR / "pilot" / "pilot_screening_ledger.csv"
        _write_csv(manifest_path, manifest)
        _write_csv(ledger_path, ledger)
    else:
        ledger = screen_dataframe(source)
        h03_count = int(ledger["H03"].sum())
        h05_count = int(ledger["H05"].sum())
        if h03_count != 86 or h05_count != 2:
            raise RuntimeError(f"Known QA regression mismatch: H03={h03_count}/86, H05={h05_count}/2")
        ledger_path = RESULTS_DIR / "bulk" / run_id / "block_screening_ledger.csv.gz"
        _write_gzip_csv(ledger_path, ledger)

    if ledger["machine_status"].isna().any() or len(ledger) != ledger["machine_status"].value_counts().sum():
        raise RuntimeError("Machine statuses are not exhaustive")
    audit_dir = _write_audit(run_id, ledger)
    database = {
        "name": database_name,
        "server": database_version,
        "transaction_read_only": read_only,
        "access": "shared/db_readonly.py",
    }
    _write_metadata(run_id, mode, database, source, ledger_path)
    result = {
        "run_id": run_id,
        "mode": mode,
        "source_rows": observed_rows,
        "source_cities": observed_cities,
        "ledger_rows": len(ledger),
        "exclude_hard": int(ledger["machine_status"].eq("exclude_hard").sum()),
        "review_scope": int(ledger["machine_status"].eq("review_scope").sum()),
        "keep_provisional": int(ledger["machine_status"].eq("keep_provisional").sum()),
        "H03": h03_count,
        "H05": h05_count,
        "ledger_sha256": _sha256(ledger_path),
        "audit_directory": str(audit_dir),
    }
    with log_path.open("a", encoding="utf-8") as log:
        log.write(json.dumps(result, ensure_ascii=False) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("pilot", "full"), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--expected-rows", type=int, default=int(RULES["expected_rows"]))
    parser.add_argument("--expected-cities", type=int, default=int(RULES["expected_cities"]))
    args = parser.parse_args()
    run(args.mode, args.run_id, args.expected_rows, args.expected_cities)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
