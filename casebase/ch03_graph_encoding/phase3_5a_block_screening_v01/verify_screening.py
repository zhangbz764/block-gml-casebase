"""Offline verification for Phase 3.5A pilot or full screening ledgers."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


PHASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PHASE_DIR / "results"

from run_phase3_5a import RULES, _sha256, _source_hashes, _write_csv, stable_csv_bytes
from screen_blocks import (
    EXPECTED_SINGLE_NULLS,
    HARD_RULES,
    METRIC_FIELDS,
    SCOPE_RULES,
    screen_dataframe,
)


def _bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    return series.astype("string").str.lower().map({"true": True, "false": False}).fillna(False).astype(bool)


def _normalize_flags(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for field in (*HARD_RULES, *SCOPE_RULES, "geom_is_null", "geom_is_empty", "geom_is_valid", "geom_area_positive"):
        if field in result:
            result[field] = _bool_series(result[field])
    return result


def single_expected_nulls_are_preserved(ledger: pd.DataFrame) -> bool:
    """Check H04 only for single-building rows whose nulls are all expected."""
    numeric = ledger.loc[:, METRIC_FIELDS].apply(pd.to_numeric, errors="coerce")
    unexpected_fields = [field for field in METRIC_FIELDS if field not in EXPECTED_SINGLE_NULLS]
    expected_only = numeric["lod1_building_count"].eq(1) & ~numeric.loc[:, unexpected_fields].isna().any(axis=1)
    return not bool(ledger.loc[expected_only, "H04"].astype(bool).any())


def pilot_recompute_matches(manifest_path: Path, ledger_path: Path) -> bool:
    """Recompute a pilot using round-trip float parsing and compare exact bytes."""
    manifest = _normalize_flags(
        pd.read_csv(manifest_path, encoding="utf-8-sig", float_precision="round_trip")
    )
    recomputed = screen_dataframe(manifest)
    return hashlib.sha256(stable_csv_bytes(recomputed)).hexdigest() == _sha256(ledger_path)


def artifact_git_policy(relative_path: Path) -> str:
    """Mirror the repository's generic Chapter 3 ignore policy."""
    name = relative_path.name
    path = relative_path.as_posix()
    if (
        name.endswith(".gz")
        or name.endswith(".html")
        or name == "run_log.txt"
        or name == "plan_review.md"
        or name.endswith("_evidence_pack.md")
        or "/bulk/" in f"/{path}"
    ):
        return "ignored"
    return "tracked"


def _write_output_inventory(audit_dir: Path) -> None:
    inventory_path = audit_dir / "output_inventory.csv"
    paths = sorted(
        path for path in RESULTS_DIR.rglob("*")
        if path.is_file() and path.name != "output_inventory.csv"
    )
    rows = []
    for path in paths:
        relative = path.relative_to(RESULTS_DIR)
        rows.append(
            {
                "relative_path": relative.as_posix(),
                "type": "csv_gzip" if path.name.endswith(".csv.gz") else path.suffix.lstrip("."),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
                "git_policy": artifact_git_policy(relative),
                "reproducible": True,
            }
        )
    _write_csv(inventory_path, pd.DataFrame(rows))


def verify(mode: str, run_id: str) -> dict[str, Any]:
    audit_dir = RESULTS_DIR / "audit" / run_id
    if mode == "pilot":
        manifest_path = RESULTS_DIR / "pilot" / "pilot_manifest.csv"
        ledger_path = RESULTS_DIR / "pilot" / "pilot_screening_ledger.csv"
        manifest = _normalize_flags(
            pd.read_csv(manifest_path, encoding="utf-8-sig", float_precision="round_trip")
        )
    else:
        manifest_path = None
        ledger_path = RESULTS_DIR / "bulk" / run_id / "block_screening_ledger.csv.gz"
        manifest = None
    ledger = _normalize_flags(
        pd.read_csv(ledger_path, encoding="utf-8-sig", float_precision="round_trip")
    )
    issues: list[dict[str, str]] = []

    def add(check: str, detail: str) -> None:
        issues.append({"check": check, "detail": detail})

    expected_rows = len(ledger) if mode == "pilot" else int(RULES["expected_rows"])
    expected_cities = int(RULES["expected_cities"])
    if len(ledger) != expected_rows:
        add("row_count", f"expected {expected_rows}, got {len(ledger)}")
    if ledger["city"].nunique(dropna=True) != expected_cities:
        add("city_count", f"expected {expected_cities}, got {ledger['city'].nunique(dropna=True)}")
    if ledger[["city", "block_id"]].duplicated().any():
        add("unique_key", "duplicate city + block_id exists")
    statuses = {"exclude_hard", "review_scope", "keep_provisional"}
    if not ledger["machine_status"].isin(statuses).all():
        add("status_domain", "unknown or missing machine status exists")
    status_counts = ledger["machine_status"].value_counts()
    if int(status_counts.sum()) != len(ledger):
        add("status_closure", "status counts do not equal ledger rows")
    if not ledger.loc[ledger["hard_flag_count"].gt(0), "machine_status"].eq("exclude_hard").all():
        add("hard_status", "hard-flagged row is not exclude_hard")
    if not ledger.loc[ledger["hard_flag_count"].eq(0) & ledger["scope_flag_count"].gt(0), "machine_status"].eq("review_scope").all():
        add("scope_status", "soft-only row is not review_scope")
    if not ledger.loc[ledger["hard_flag_count"].eq(0) & ledger["scope_flag_count"].eq(0), "machine_status"].eq("keep_provisional").all():
        add("keep_status", "unflagged row is not keep_provisional")
    if int(ledger["H03"].sum()) != 86:
        add("known_H03", f"expected 86, got {int(ledger['H03'].sum())}")
    if int(ledger["H05"].sum()) != 2:
        add("known_H05", f"expected 2, got {int(ledger['H05'].sum())}")
    if not single_expected_nulls_are_preserved(ledger):
        add("single_building_expected_nulls", "a single-building row with only expected nulls was flagged H04")
    if any(ledger[field].dtype != bool for field in (*HARD_RULES, *SCOPE_RULES)):
        add("flag_types", "a rule flag is not boolean")

    saved_flow = pd.read_csv(audit_dir / "filter_flow_summary.csv", encoding="utf-8-sig")
    saved_city = pd.read_csv(audit_dir / "filter_summary_by_city.csv", encoding="utf-8-sig")
    if int(saved_flow.iloc[-1]["cumulative_excluded_count"]) != int(status_counts.get("exclude_hard", 0)):
        add("flow_closure", "sequential hard flow does not equal exclude_hard count")
    if int(saved_city["input_count"].sum()) != len(ledger):
        add("city_closure", "city inputs do not equal ledger rows")

    reproducible = None
    if mode == "pilot" and manifest is not None:
        reproducible = pilot_recompute_matches(manifest_path, ledger_path)
        if not reproducible:
            add("deterministic_recompute", "pilot ledger hash changed on recomputation")
        baseline_counts = manifest.loc[
            manifest["selection_reasons"].astype("string").str.contains("city_head10", na=False)
        ].groupby("city").size()
        if len(baseline_counts) != expected_cities or not baseline_counts.eq(10).all():
            add("pilot_city_baseline", "pilot does not contain exactly 10 baseline rows per city")

    result = {
        "experiment_id": RULES["experiment_id"],
        "rule_version": RULES["rule_version"],
        "run_id": run_id,
        "mode": mode,
        "status": "PILOT_VERIFIED" if mode == "pilot" and not issues else ("VERIFIED" if not issues else "FAILED"),
        "passed": not issues,
        "ledger_rows": len(ledger),
        "city_count": int(ledger["city"].nunique(dropna=True)),
        "machine_status_counts": {key: int(status_counts.get(key, 0)) for key in sorted(statuses)},
        "H03_count": int(ledger["H03"].sum()),
        "H05_count": int(ledger["H05"].sum()),
        "deterministic_recompute_passed": reproducible,
        "ledger_sha256": _sha256(ledger_path),
        "issue_count": len(issues),
        "issues": issues,
        "full_run_executed": mode == "full",
    }
    audit_dir.mkdir(parents=True, exist_ok=True)
    (audit_dir / "verification_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if mode == "pilot":
        (RESULTS_DIR / "pilot" / "pilot_verification.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    freeze_path = RESULTS_DIR / "freeze.json"
    if freeze_path.exists():
        freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
        freeze["source_sha256"] = _source_hashes()
        freeze["verification_sha256"] = _sha256(
            RESULTS_DIR / "pilot" / "pilot_verification.json"
            if mode == "pilot"
            else audit_dir / "verification_summary.json"
        )
        freeze_path.write_text(json.dumps(freeze, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_output_inventory(audit_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("pilot", "full"), required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    return 0 if verify(args.mode, args.run_id)["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
