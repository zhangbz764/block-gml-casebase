"""Build the deterministic S02-only and S04-only Phase 3.5A review set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


PHASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PHASE_DIR / "results"

from run_phase3_5a import _write_csv, validate_run_id


HARD_RULES = tuple(f"H{i:02d}" for i in range(1, 6))
SOFT_RULES = tuple(f"S{i:02d}" for i in range(1, 7))
EXPECTED_POOL_COUNTS = {"S02_only": 1049, "S04_only": 1184}
SOLO_RULES = {
    "S02_only": {"target": "S02", "metric": "elongation", "extreme_ascending": True},
    "S04_only": {"target": "S04", "metric": "lod1_building_count", "extreme_ascending": False},
}


def _coerce_rule_flags(ledger: pd.DataFrame) -> pd.DataFrame:
    frame = ledger.copy()
    missing = [column for column in (*HARD_RULES, *SOFT_RULES, "hard_flag_count") if column not in frame]
    if missing:
        raise ValueError(f"Ledger is missing required columns: {', '.join(missing)}")
    for column in (*HARD_RULES, *SOFT_RULES):
        frame[column] = (
            frame[column]
            .astype("string")
            .str.lower()
            .map({"true": True, "false": False})
            .fillna(False)
            .astype(bool)
        )
    frame["hard_flag_count"] = pd.to_numeric(frame["hard_flag_count"], errors="coerce")
    return frame


def build_solo_pools(ledger: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return strict S02-only and S04-only pools, retaining all ledger fields."""
    frame = _coerce_rule_flags(ledger)
    pools: dict[str, pd.DataFrame] = {}
    for solo_rule, config in SOLO_RULES.items():
        target = str(config["target"])
        other_soft = [rule for rule in SOFT_RULES if rule != target]
        mask = frame[target] & ~frame[other_soft].any(axis=1) & frame["hard_flag_count"].eq(0)
        pools[solo_rule] = frame.loc[mask].copy()
    return pools


def _ordered(pool: pd.DataFrame, metric: str, ascending: bool) -> pd.DataFrame:
    return pool.sort_values(
        [metric, "city", "block_id"],
        ascending=[ascending, True, True],
        kind="mergesort",
    )


def _city_rotation(pool: pd.DataFrame) -> pd.DataFrame:
    ordered = pool.sort_values(["city", "block_id"], kind="mergesort")
    return ordered.groupby("city", sort=True, as_index=False, dropna=False).head(1)


def _select_role(
    candidates: pd.DataFrame,
    fallback: pd.DataFrame,
    selected_keys: set[tuple[str, str]],
    *,
    limit: int,
) -> list[pd.Series]:
    chosen: list[pd.Series] = []
    for source in (candidates, fallback):
        for _, row in source.iterrows():
            if len(chosen) >= limit:
                return chosen
            key = (str(row["city"]), str(row["block_id"]))
            if key in selected_keys:
                continue
            selected_keys.add(key)
            chosen.append(row.copy())
    return chosen


def build_solo_manifest(
    ledger: pd.DataFrame,
    *,
    per_rule: int = 12,
    source_run_id: str = "full_v01",
) -> pd.DataFrame:
    """Select deterministic extreme, near-threshold, and city-rotation cases."""
    if per_rule < 3:
        raise ValueError("per_rule must be at least 3")
    validate_run_id(source_run_id)
    pools = build_solo_pools(ledger)
    base_role_count, remainder = divmod(per_rule, 3)
    role_limits = {
        "extreme": base_role_count,
        "near_threshold": base_role_count,
        "city_rotation": base_role_count + remainder,
    }
    records: list[dict[str, Any]] = []

    for solo_rule, config in SOLO_RULES.items():
        pool = pools[solo_rule]
        if len(pool) < per_rule:
            raise ValueError(f"{solo_rule} has only {len(pool)} candidates; cannot select {per_rule}")
        metric = str(config["metric"])
        extreme = _ordered(pool, metric, bool(config["extreme_ascending"]))
        near_threshold = _ordered(pool, metric, not bool(config["extreme_ascending"]))
        fixed_fallback = pool.sort_values(["city", "block_id"], kind="mergesort")
        selected_keys: set[tuple[str, str]] = set()
        role_sources = (
            ("extreme", extreme),
            ("near_threshold", near_threshold),
        )
        for role, candidates in role_sources:
            chosen = _select_role(
                candidates,
                fixed_fallback,
                selected_keys,
                limit=role_limits[role],
            )
            if len(chosen) != role_limits[role]:
                raise RuntimeError(f"Could not fill {solo_rule} role {role}")
            for row in chosen:
                record = row.to_dict()
                record.update(
                    {
                        "solo_rule": solo_rule,
                        "review_sample_roles": role,
                        "review_case_type": "solo_flag_review",
                        "source_run_id": source_run_id,
                    }
                )
                records.append(record)

        remaining = pool.loc[
            ~pool.apply(
                lambda row: (str(row["city"]), str(row["block_id"])) in selected_keys,
                axis=1,
            )
        ]
        role = "city_rotation"
        chosen = _select_role(
            _city_rotation(remaining),
            fixed_fallback,
            selected_keys,
            limit=role_limits[role],
        )
        if len(chosen) != role_limits[role]:
            raise RuntimeError(f"Could not fill {solo_rule} role {role}")
        for row in chosen:
            record = row.to_dict()
            record.update(
                {
                    "solo_rule": solo_rule,
                    "review_sample_roles": role,
                    "review_case_type": "solo_flag_review",
                    "source_run_id": source_run_id,
                }
            )
            records.append(record)

    manifest = pd.DataFrame(records)
    if manifest[["city", "block_id"]].duplicated().any():
        raise RuntimeError("Solo review manifest contains duplicate block keys")
    return manifest.reset_index(drop=True)


def verification_summary(
    pool_counts: dict[str, int],
    manifest: pd.DataFrame,
    *,
    per_rule: int,
) -> dict[str, Any]:
    sample_counts = {
        key: int(value)
        for key, value in manifest.groupby("solo_rule", sort=True).size().to_dict().items()
    }
    role_counts = {
        key: int(value)
        for key, value in manifest.groupby("review_sample_roles", sort=True).size().to_dict().items()
    }
    unique_count = int(manifest[["city", "block_id"]].drop_duplicates().shape[0])
    expected_samples = {solo_rule: per_rule for solo_rule in SOLO_RULES}
    passed = (
        pool_counts == EXPECTED_POOL_COUNTS
        and sample_counts == expected_samples
        and unique_count == per_rule * len(SOLO_RULES)
        and set(role_counts) == {"extreme", "near_threshold", "city_rotation"}
    )
    return {
        "candidate_pool_counts": pool_counts,
        "expected_candidate_pool_counts": EXPECTED_POOL_COUNTS,
        "sample_counts": sample_counts,
        "unique_block_count": unique_count,
        "role_counts": role_counts,
        "passed": bool(passed),
    }


def output_paths(
    results_dir: Path,
    *,
    source_run_id: str,
    output_run_id: str,
) -> dict[str, Path]:
    validate_run_id(source_run_id)
    validate_run_id(output_run_id)
    if source_run_id == output_run_id:
        raise ValueError("output_run_id must differ from source_run_id")
    audit_dir = results_dir / "audit" / output_run_id
    return {
        "manifest": audit_dir / "review_manifest.csv",
        "pool_summary": audit_dir / "solo_pool_summary.csv",
        "verification": audit_dir / "verification_summary.json",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run-id", required=True)
    parser.add_argument("--output-run-id", required=True)
    parser.add_argument("--per-rule", type=int, default=12)
    args = parser.parse_args()

    paths = output_paths(
        RESULTS_DIR,
        source_run_id=args.source_run_id,
        output_run_id=args.output_run_id,
    )
    audit_dir = paths["manifest"].parent
    if audit_dir.exists():
        raise FileExistsError(f"Refusing to overwrite solo review output: {audit_dir}")

    ledger_path = RESULTS_DIR / "bulk" / args.source_run_id / "block_screening_ledger.csv.gz"
    if not ledger_path.exists():
        raise FileNotFoundError(f"Source screening ledger not found: {ledger_path}")
    ledger = pd.read_csv(ledger_path, encoding="utf-8-sig", low_memory=False)
    pools = build_solo_pools(ledger)
    pool_counts = {name: len(pool) for name, pool in pools.items()}
    if pool_counts != EXPECTED_POOL_COUNTS:
        raise RuntimeError(
            "Solo candidate pool counts do not match frozen expectations: "
            + json.dumps(
                {"actual": pool_counts, "expected": EXPECTED_POOL_COUNTS},
                ensure_ascii=False,
                sort_keys=True,
            )
        )

    manifest = build_solo_manifest(
        ledger,
        per_rule=args.per_rule,
        source_run_id=args.source_run_id,
    )
    verification = verification_summary(pool_counts, manifest, per_rule=args.per_rule)
    if not verification["passed"]:
        raise RuntimeError("Solo review verification failed: " + json.dumps(verification, ensure_ascii=False))

    pool_summary = pd.DataFrame(
        [
            {
                "solo_rule": solo_rule,
                "candidate_pool_count": pool_counts[solo_rule],
                "expected_candidate_pool_count": EXPECTED_POOL_COUNTS[solo_rule],
                "sample_count": int((manifest["solo_rule"] == solo_rule).sum()),
                "count_matches": pool_counts[solo_rule] == EXPECTED_POOL_COUNTS[solo_rule],
            }
            for solo_rule in SOLO_RULES
        ]
    )
    _write_csv(paths["manifest"], manifest)
    _write_csv(paths["pool_summary"], pool_summary)
    paths["verification"].write_text(
        json.dumps(verification, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "source_run_id": args.source_run_id,
                "output_run_id": args.output_run_id,
                "candidate_pool_counts": pool_counts,
                "sample_count": len(manifest),
                "output": str(audit_dir),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
