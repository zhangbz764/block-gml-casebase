"""Build a bounded, deterministic review manifest from a full screening ledger."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd


PHASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PHASE_DIR / "results"

from run_phase3_5a import _write_csv, validate_run_id
from screen_blocks import SCOPE_RULES


SCOPE_ORDER = {
    "S01": (("area_m2", False),),
    "S02": (("elongation", True),),
    "S03": (("compactness", True),),
    "S04": (("lod1_building_count", False),),
    "S05": (("bcr", True),),
    "S06": (("area_m2", False), ("bcr", True)),
}
THRESHOLD_KEY = {
    "S01": "area_m2_p99",
    "S02": "elongation_p01",
    "S03": "compactness_p01",
    "S04": "lod1_building_count_p99",
    "S05": "bcr_p01",
}


def _sort_extreme(pool: pd.DataFrame, rule: str) -> pd.DataFrame:
    fields = [item[0] for item in SCOPE_ORDER[rule]]
    ascending = [item[1] for item in SCOPE_ORDER[rule]]
    return pool.sort_values(
        [*fields, "city", "block_id"],
        ascending=[*ascending, True, True],
        kind="mergesort",
    )


def _sort_boundary(pool: pd.DataFrame, rule: str, thresholds: Mapping[str, float]) -> pd.DataFrame:
    scored = pool.copy()
    if rule == "S06":
        area_scale = max(abs(float(thresholds["area_m2_p99"])), 1e-12)
        bcr_scale = max(abs(float(thresholds["bcr_p01"])), 1e-12)
        scored["_distance"] = (
            (scored["area_m2"] - thresholds["area_m2_p99"]).abs() / area_scale
            + (scored["bcr"] - thresholds["bcr_p01"]).abs() / bcr_scale
        )
    else:
        field = SCOPE_ORDER[rule][0][0]
        scored["_distance"] = (scored[field] - thresholds[THRESHOLD_KEY[rule]]).abs()
    return scored.sort_values(["_distance", "city", "block_id"], kind="mergesort")


def _city_rotation(pool: pd.DataFrame) -> pd.DataFrame:
    ordered = pool.sort_values(["city", "block_id"], kind="mergesort")
    first_per_city = ordered.groupby("city", sort=True, as_index=False).head(1)
    return first_per_city.sort_values(["city", "block_id"], kind="mergesort")


def build_review_manifest(
    ledger: pd.DataFrame,
    thresholds: Mapping[str, float],
    *,
    max_cases: int = 72,
) -> pd.DataFrame:
    """Select soft-tail examples and a few hard-rule checks without duplicates."""
    if max_cases < 1:
        raise ValueError("max_cases must be positive")
    ordered = ledger.copy().sort_values(["city", "block_id"], kind="mergesort").reset_index(drop=True)
    selected: dict[tuple[str, str], dict[str, Any]] = {}

    def add_rows(rows: pd.DataFrame, role: str, limit: int) -> None:
        added_for_role = 0
        for _, row in rows.iterrows():
            if added_for_role >= limit:
                break
            key = (str(row["city"]), str(row["block_id"]))
            if key not in selected:
                if len(selected) >= max_cases:
                    continue
                selected[key] = {"row": row.copy(), "roles": []}
            if role not in selected[key]["roles"]:
                selected[key]["roles"].append(role)
                added_for_role += 1

    for rule in ("H03", "H04", "H05"):
        pool = ordered.loc[ordered[rule].astype(bool)].sort_values(["city", "block_id"], kind="mergesort")
        add_rows(pool, f"{rule}_hard_check", 4)

    for rule in SCOPE_RULES:
        pool = ordered.loc[ordered[rule].astype(bool) & ordered["hard_flag_count"].eq(0)].copy()
        already_for_rule: set[tuple[str, str]] = set()
        categories = (
            ("extreme", _sort_extreme(pool, rule)),
            ("threshold", _sort_boundary(pool, rule, thresholds)),
            ("city_rotation", _city_rotation(pool)),
        )
        for category, candidates in categories:
            available = candidates.loc[
                ~candidates.apply(lambda row: (str(row["city"]), str(row["block_id"])) in already_for_rule, axis=1)
            ]
            before_roles = {
                key for key, item in selected.items()
                if any(role.startswith(f"{rule}_") for role in item["roles"])
            }
            add_rows(available, f"{rule}_{category}", 4)
            after_roles = {
                key for key, item in selected.items()
                if any(role.startswith(f"{rule}_") for role in item["roles"])
            }
            already_for_rule.update(after_roles - before_roles)

    records = []
    for item in selected.values():
        record = item["row"].to_dict()
        record["review_sample_roles"] = ";".join(item["roles"])
        record["review_case_type"] = (
            "hard_check" if all(role.endswith("hard_check") for role in item["roles"])
            else "soft_review"
        )
        record["review_thresholds"] = json.dumps(dict(thresholds), ensure_ascii=False, sort_keys=True)
        for field in ("manual_decision", "manual_reason", "manual_notes", "reviewed_at"):
            record[field] = ""
        records.append(record)
    if not records:
        return pd.DataFrame(columns=[*ledger.columns, "review_sample_roles", "review_case_type", "review_thresholds"])
    return pd.DataFrame(records).sort_values(["city", "block_id"], kind="mergesort").reset_index(drop=True)


def _load_thresholds(path: Path) -> dict[str, float]:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    result = {}
    for row in frame.itertuples(index=False):
        suffix = "p01" if float(row.quantile) == 0.01 else "p99"
        result[f"{row.metric}_{suffix}"] = float(row.threshold)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--max-cases", type=int, default=72)
    args = parser.parse_args()
    validate_run_id(args.run_id)
    ledger_path = RESULTS_DIR / "bulk" / args.run_id / "block_screening_ledger.csv.gz"
    audit_dir = RESULTS_DIR / "audit" / args.run_id
    output_path = audit_dir / "review_manifest.csv"
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing review manifest: {output_path}")
    ledger = pd.read_csv(ledger_path, encoding="utf-8-sig")
    for rule in (*[f"H{i:02d}" for i in range(1, 6)], *SCOPE_RULES):
        ledger[rule] = ledger[rule].astype("string").str.lower().map({"true": True, "false": False}).fillna(False)
    thresholds = _load_thresholds(audit_dir / "threshold_summary.csv")
    manifest = build_review_manifest(ledger, thresholds, max_cases=args.max_cases)
    _write_csv(output_path, manifest)
    print(json.dumps({"run_id": args.run_id, "review_case_count": len(manifest), "output": str(output_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
