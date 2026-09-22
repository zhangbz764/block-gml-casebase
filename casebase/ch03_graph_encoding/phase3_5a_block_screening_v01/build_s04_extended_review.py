"""Build the deterministic 60-case extended S04-only review sample."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


PHASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PHASE_DIR / "results"

from build_solo_flag_review import build_solo_pools
from run_phase3_5a import _write_csv, validate_run_id


EXPECTED_S04_POOL_SIZE = 1184
EXPECTED_S04_CITY_COUNT = 20
PRIOR_SAMPLE_SIZE = 12
GRID_SAMPLE_SIZE = 32
BAND_LABELS = ("Q1", "Q2", "Q3", "Q4")


def _key(row: pd.Series) -> tuple[str, str]:
    return str(row["city"]), str(row["block_id"])


def _rank_band(values: pd.DataFrame, metric: str) -> pd.Series:
    ordered = values.sort_values([metric, "city", "block_id"], kind="mergesort")
    count = len(ordered)
    if count < 4:
        raise ValueError("At least four candidates are required for quartile bands")
    labels = [BAND_LABELS[min(3, position * 4 // count)] for position in range(count)]
    result = pd.Series(labels, index=ordered.index, dtype="string")
    return result.reindex(values.index)


def add_rank_bands(pool: pd.DataFrame) -> pd.DataFrame:
    """Assign stable ordinal quartiles within the full S04-only pool."""
    frame = pool.copy()
    frame["count_band"] = _rank_band(frame, "lod1_building_count")
    frame["area_band"] = _rank_band(frame, "area_m2")
    return frame


def _append_record(records: list[dict[str, Any]], row: pd.Series, origin: str) -> None:
    record = row.to_dict()
    record["sample_origin"] = origin
    records.append(record)


def build_extended_manifest(
    ledger: pd.DataFrame,
    prior_manifest: pd.DataFrame,
    *,
    new_sample_size: int = 48,
    seed: int = 20260922,
) -> pd.DataFrame:
    """Retain the prior 12 and add a stratified, city-complete S04-only sample."""
    if new_sample_size < GRID_SAMPLE_SIZE:
        raise ValueError(f"new_sample_size must be at least {GRID_SAMPLE_SIZE}")
    pool = add_rank_bands(build_solo_pools(ledger)["S04_only"])
    if pool[["city", "block_id"]].duplicated().any():
        raise ValueError("S04-only pool contains duplicate block keys")

    prior = prior_manifest.copy()
    if "solo_rule" in prior:
        prior = prior.loc[prior["solo_rule"].eq("S04_only")].copy()
    if len(prior) != PRIOR_SAMPLE_SIZE:
        raise ValueError(f"Prior manifest must contain exactly {PRIOR_SAMPLE_SIZE} S04-only cases")
    if prior[["city", "block_id"]].duplicated().any():
        raise ValueError("Prior S04-only sample contains duplicate block keys")

    pool_by_key = pool.set_index(["city", "block_id"], drop=False)
    prior_keys = list(prior[["city", "block_id"]].astype(str).itertuples(index=False, name=None))
    missing_prior = [key for key in prior_keys if key not in pool_by_key.index]
    if missing_prior:
        raise ValueError(f"Prior S04-only cases are missing from the frozen pool: {missing_prior[:3]}")

    records: list[dict[str, Any]] = []
    selected_keys: set[tuple[str, str]] = set()
    for key in prior_keys:
        row = pool_by_key.loc[key]
        _append_record(records, row, "prior_12")
        selected_keys.add(key)

    remaining = pool.loc[~pool.apply(lambda row: _key(row) in selected_keys, axis=1)].copy()
    rng = random.Random(seed)
    for count_band in BAND_LABELS:
        for area_band in BAND_LABELS:
            cell = remaining.loc[
                remaining["count_band"].eq(count_band) & remaining["area_band"].eq(area_band)
            ].sort_values(["city", "block_id"], kind="mergesort")
            if len(cell) < 2:
                raise ValueError(f"Metric grid cell {count_band} x {area_band} has fewer than two cases")
            positions = sorted(rng.sample(range(len(cell)), 2))
            for position in positions:
                row = cell.iloc[position]
                key = _key(row)
                _append_record(records, row, "metric_grid")
                selected_keys.add(key)

    candidate_cities = sorted(pool["city"].astype(str).unique())
    selected_cities = {str(record["city"]) for record in records}
    missing_cities = [city for city in candidate_cities if city not in selected_cities]
    slots_left = PRIOR_SAMPLE_SIZE + new_sample_size - len(records)
    if len(missing_cities) > slots_left:
        raise ValueError("Requested sample size cannot cover every candidate city")
    for city in missing_cities:
        candidates = pool.loc[
            pool["city"].astype(str).eq(city)
            & ~pool.apply(lambda row: _key(row) in selected_keys, axis=1)
        ].sort_values(["block_id"], kind="mergesort")
        if candidates.empty:
            raise RuntimeError(f"No unselected case remains for city {city}")
        row = candidates.iloc[0]
        _append_record(records, row, "city_coverage")
        selected_keys.add(_key(row))

    target_total = PRIOR_SAMPLE_SIZE + new_sample_size
    while len(records) < target_total:
        city_counts = Counter(str(record["city"]) for record in records)
        candidates = pool.loc[
            ~pool.apply(lambda row: _key(row) in selected_keys, axis=1)
        ].copy()
        if candidates.empty:
            raise RuntimeError("S04-only pool was exhausted before the requested sample size")
        candidates["_selected_city_count"] = candidates["city"].astype(str).map(city_counts).fillna(0)
        candidates = candidates.sort_values(
            ["_selected_city_count", "city", "block_id"],
            kind="mergesort",
        )
        row = candidates.iloc[0].drop(labels=["_selected_city_count"])
        _append_record(records, row, "city_balance")
        selected_keys.add(_key(row))

    manifest = pd.DataFrame(records).reset_index(drop=True)
    manifest["extended_case_no"] = range(1, len(manifest) + 1)
    if manifest[["city", "block_id"]].duplicated().any():
        raise RuntimeError("Extended S04 manifest contains duplicate block keys")
    return manifest


def build_manual_review(manifest: pd.DataFrame) -> pd.DataFrame:
    manual = manifest[
        ["extended_case_no", "city", "block_id", "sample_origin", "count_band", "area_band"]
    ].copy()
    manual["manual_decision"] = ""
    manual["manual_reason"] = ""
    manual["manual_notes"] = ""
    return manual.sort_values("extended_case_no", kind="mergesort").reset_index(drop=True)


def build_sampling_summary(manifest: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add(dimension: str, grouped: pd.Series) -> None:
        for key, count in grouped.items():
            if isinstance(key, tuple):
                value, secondary = key
            else:
                value, secondary = key, ""
            rows.append(
                {
                    "dimension": dimension,
                    "value": value,
                    "secondary_value": secondary,
                    "sample_count": int(count),
                }
            )

    add("city", manifest.groupby("city", sort=True).size())
    add("count_band", manifest.groupby("count_band", sort=True).size())
    add("area_band", manifest.groupby("area_band", sort=True).size())
    add("metric_grid", manifest.groupby(["count_band", "area_band"], sort=True).size())
    add("sample_origin", manifest.groupby("sample_origin", sort=True).size())
    return pd.DataFrame(rows)


def verification_summary(
    *,
    pool_size: int,
    candidate_city_count: int,
    manifest: pd.DataFrame,
    prior_keys: set[tuple[str, str]],
    new_sample_size: int,
) -> dict[str, Any]:
    keys = set(manifest[["city", "block_id"]].astype(str).itertuples(index=False, name=None))
    grid = manifest.loc[manifest["sample_origin"].eq("metric_grid")]
    grid_counts = grid.groupby(["count_band", "area_band"], sort=True).size()
    origin_counts = {
        key: int(value)
        for key, value in manifest.groupby("sample_origin", sort=True).size().to_dict().items()
    }
    summary = {
        "candidate_pool_count": int(pool_size),
        "expected_candidate_pool_count": EXPECTED_S04_POOL_SIZE,
        "candidate_city_count": int(candidate_city_count),
        "sampled_city_count": int(manifest["city"].nunique()),
        "sample_count": int(len(manifest)),
        "unique_block_count": int(len(keys)),
        "prior_sample_count": int(manifest["sample_origin"].eq("prior_12").sum()),
        "prior_cases_retained": bool(prior_keys.issubset(keys)),
        "new_sample_count": int(manifest["sample_origin"].ne("prior_12").sum()),
        "metric_grid_sample_count": int(len(grid)),
        "metric_grid_cell_count": int(len(grid_counts)),
        "metric_grid_min_per_cell": int(grid_counts.min()) if not grid_counts.empty else 0,
        "sample_origin_counts": origin_counts,
    }
    summary["passed"] = bool(
        summary["candidate_pool_count"] == EXPECTED_S04_POOL_SIZE
        and summary["candidate_city_count"] == EXPECTED_S04_CITY_COUNT
        and summary["sampled_city_count"] == EXPECTED_S04_CITY_COUNT
        and summary["sample_count"] == PRIOR_SAMPLE_SIZE + new_sample_size
        and summary["unique_block_count"] == summary["sample_count"]
        and summary["prior_sample_count"] == PRIOR_SAMPLE_SIZE
        and summary["prior_cases_retained"]
        and summary["new_sample_count"] == new_sample_size
        and summary["metric_grid_sample_count"] == GRID_SAMPLE_SIZE
        and summary["metric_grid_cell_count"] == 16
        and summary["metric_grid_min_per_cell"] >= 2
    )
    return summary


def output_paths(
    results_dir: Path,
    *,
    source_run_id: str,
    prior_run_id: str,
    output_run_id: str,
) -> dict[str, Path]:
    for run_id in (source_run_id, prior_run_id, output_run_id):
        validate_run_id(run_id)
    if output_run_id in {source_run_id, prior_run_id}:
        raise ValueError("output_run_id must differ from source_run_id and prior_run_id")
    audit_dir = results_dir / "audit" / output_run_id
    return {
        "manifest": audit_dir / "review_manifest.csv",
        "sampling_summary": audit_dir / "sampling_summary.csv",
        "manual_review": audit_dir / "manual_review.csv",
        "verification": audit_dir / "verification_summary.json",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run-id", required=True)
    parser.add_argument("--prior-run-id", required=True)
    parser.add_argument("--output-run-id", required=True)
    parser.add_argument("--new-sample-size", type=int, default=48)
    parser.add_argument("--seed", type=int, default=20260922)
    args = parser.parse_args()

    paths = output_paths(
        RESULTS_DIR,
        source_run_id=args.source_run_id,
        prior_run_id=args.prior_run_id,
        output_run_id=args.output_run_id,
    )
    audit_dir = paths["manifest"].parent
    visual_dir = RESULTS_DIR / "visual_review" / args.output_run_id
    collisions = [path for path in (audit_dir, visual_dir) if path.exists()]
    if collisions:
        raise FileExistsError(f"Refusing to overwrite existing run directories: {collisions}")

    ledger_path = RESULTS_DIR / "bulk" / args.source_run_id / "block_screening_ledger.csv.gz"
    prior_path = RESULTS_DIR / "audit" / args.prior_run_id / "review_manifest.csv"
    ledger = pd.read_csv(ledger_path, encoding="utf-8-sig", low_memory=False)
    prior = pd.read_csv(prior_path, encoding="utf-8-sig", keep_default_na=False)
    s04_pool = build_solo_pools(ledger)["S04_only"]
    if len(s04_pool) != EXPECTED_S04_POOL_SIZE:
        raise RuntimeError(
            f"S04-only pool count is {len(s04_pool)}, expected {EXPECTED_S04_POOL_SIZE}"
        )
    manifest = build_extended_manifest(
        ledger,
        prior,
        new_sample_size=args.new_sample_size,
        seed=args.seed,
    )
    prior_s04 = prior.loc[prior["solo_rule"].eq("S04_only")]
    prior_keys = set(prior_s04[["city", "block_id"]].astype(str).itertuples(index=False, name=None))
    verification = verification_summary(
        pool_size=len(s04_pool),
        candidate_city_count=int(s04_pool["city"].nunique()),
        manifest=manifest,
        prior_keys=prior_keys,
        new_sample_size=args.new_sample_size,
    )
    if not verification["passed"]:
        raise RuntimeError("Extended S04 review verification failed: " + json.dumps(verification, ensure_ascii=False))

    _write_csv(paths["manifest"], manifest)
    _write_csv(paths["sampling_summary"], build_sampling_summary(manifest))
    _write_csv(paths["manual_review"], build_manual_review(manifest))
    paths["verification"].write_text(
        json.dumps(verification, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "source_run_id": args.source_run_id,
                "prior_run_id": args.prior_run_id,
                "output_run_id": args.output_run_id,
                "sample_count": len(manifest),
                "sampled_city_count": int(manifest["city"].nunique()),
                "output": str(audit_dir),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
