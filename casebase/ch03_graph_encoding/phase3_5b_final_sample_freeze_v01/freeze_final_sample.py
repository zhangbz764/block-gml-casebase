"""Freeze the final Phase 3.5B block sample and graph-encoding entry list."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import platform
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


PHASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PHASE_DIR / "results"
FINAL_RULE_VERSION = "0.2"
GRAPH_SCALE_QUANTILE = 0.995
DEFAULT_SOURCE_RUN_ID = "full_v01"
DEFAULT_RUN_ID = "final_v01"
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
HARD_RULES = tuple(f"H{i:02d}" for i in range(1, 6))
SOFT_RULES = tuple(f"S{i:02d}" for i in range(1, 7))
GRAPH_MANIFEST_COLUMNS = [
    "city",
    "block_id",
    "lod1_building_count",
    "area_m2",
    "bcr",
    "final_rule_version",
    "graph_encoding_eligible",
    "source_run_id",
]


def stable_csv_bytes(frame: pd.DataFrame) -> bytes:
    text = frame.to_csv(index=False, lineterminator="\n", float_format="%.15g")
    return b"\xef\xbb\xbf" + text.encode("utf-8")


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(stable_csv_bytes(frame))


def write_gzip_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            compressed.write(stable_csv_bytes(frame))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return (
        series.astype("string")
        .str.lower()
        .map({"true": True, "false": False, "1": True, "0": False})
        .fillna(False)
        .astype(bool)
    )


def _validate_rule_input(ledger: pd.DataFrame) -> None:
    required = {
        "city", "block_id", "lod1_building_count", "area_m2", "bcr",
        "hard_flag_count", "hard_reasons", "primary_hard_reason",
        *HARD_RULES, *SOFT_RULES,
    }
    missing = sorted(required - set(ledger.columns))
    if missing:
        raise ValueError(f"Final screening input is missing columns: {', '.join(missing)}")
    if ledger[["city", "block_id"]].isna().any(axis=None):
        raise ValueError("Final screening input contains missing city or block_id")
    if ledger[["city", "block_id"]].duplicated().any():
        raise ValueError("Final screening input contains duplicate city + block_id keys")


def compute_p995_threshold(ledger: pd.DataFrame) -> float:
    """Compute global P99.5 building count from rows with no H01-H05 hit."""
    if "hard_flag_count" not in ledger or "lod1_building_count" not in ledger:
        raise ValueError("Ledger requires hard_flag_count and lod1_building_count")
    hard_count = pd.to_numeric(ledger["hard_flag_count"], errors="coerce")
    building_count = pd.to_numeric(ledger["lod1_building_count"], errors="coerce")
    population = building_count.loc[hard_count.eq(0)].dropna()
    if population.empty:
        raise ValueError("No hard-rule-eligible rows are available for P99.5")
    return float(population.quantile(GRAPH_SCALE_QUANTILE))


def _split_reasons(value: Any) -> list[str]:
    if value is None or pd.isna(value):
        return []
    return [item.strip() for item in str(value).split(";") if item.strip()]


def _deduplicate(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))


def apply_final_rules(ledger: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Apply frozen v0.2 exclusions while retaining all original H/S flags."""
    _validate_rule_input(ledger)
    result = ledger.copy()
    for column in (*HARD_RULES, *SOFT_RULES):
        result[column] = _bool_series(result[column])
    result["hard_flag_count"] = pd.to_numeric(result["hard_flag_count"], errors="raise").astype(int)
    result["lod1_building_count"] = pd.to_numeric(result["lod1_building_count"], errors="raise")
    hard = result["hard_flag_count"].gt(0)
    scale = ~hard & result["lod1_building_count"].ge(float(threshold))
    excluded = hard | result["S01"] | result["S03"] | result["S05"] | scale

    primary_reasons: list[str] = []
    all_reasons: list[str] = []
    columns = [
        "hard_flag_count", "hard_reasons", "primary_hard_reason",
        "S01", "S03", "S05", "S06", "lod1_building_count",
    ]
    for hard_count, hard_reasons, primary_hard, s01, s03, s05, s06, building_count in result[columns].itertuples(index=False, name=None):
        reasons: list[str] = []
        primary = ""
        if int(hard_count) > 0:
            primary_candidates = _split_reasons(primary_hard)
            hard_reason_list = _split_reasons(hard_reasons)
            primary = primary_candidates[0] if primary_candidates else (hard_reason_list[0] if hard_reason_list else "hard_rule_exclusion")
            reasons.extend([primary, *hard_reason_list])
        if bool(s01):
            primary = primary or "area_upper_tail_p99"
            reasons.append("area_upper_tail_p99")
        if bool(s03):
            primary = primary or "compactness_lower_tail_p01"
            reasons.append("compactness_lower_tail_p01")
        if bool(s05):
            primary = primary or "bcr_lower_tail_p01"
            reasons.append("bcr_lower_tail_p01")
        if int(hard_count) == 0 and float(building_count) >= float(threshold):
            primary = primary or "building_count_upper_tail_p995"
            reasons.append("building_count_upper_tail_p995")
        if bool(s06):
            reasons.append("large_sparse_combination")
        primary_reasons.append(primary)
        all_reasons.append(";".join(_deduplicate(reasons)))

    result["final_rule_version"] = FINAL_RULE_VERSION
    result["building_count_p995_threshold"] = float(threshold)
    result["exclude_graph_scale_p995"] = scale.astype(bool)
    result["final_decision"] = excluded.map({True: "exclude", False: "eligible"})
    result["graph_encoding_eligible"] = (~excluded).astype(bool)
    result["primary_final_reason"] = primary_reasons
    result["final_exclusion_reasons"] = all_reasons
    return result.sort_values(["city", "block_id"], kind="mergesort").reset_index(drop=True)


def build_graph_manifest(ledger: pd.DataFrame) -> pd.DataFrame:
    """Return the unique, minimal entry list for downstream graph encoding."""
    eligible = ledger.loc[ledger["final_decision"].eq("eligible")].copy()
    if eligible[["city", "block_id"]].duplicated().any():
        raise ValueError("Eligible graph manifest contains duplicate block keys")
    if "source_run_id" not in eligible:
        eligible["source_run_id"] = DEFAULT_SOURCE_RUN_ID
    manifest = eligible[GRAPH_MANIFEST_COLUMNS].copy()
    return manifest.sort_values(["city", "block_id"], kind="mergesort").reset_index(drop=True)


def _summary_flow(ledger: pd.DataFrame) -> pd.DataFrame:
    input_count = len(ledger)
    hard = ledger["hard_flag_count"].gt(0)
    rules = [
        ("H01-H05", "any_hard_rule", hard, True),
        ("S01", "area_upper_tail_p99", ledger["S01"], True),
        ("S03", "compactness_lower_tail_p01", ledger["S03"], True),
        ("S05", "bcr_lower_tail_p01", ledger["S05"], True),
        ("P995", "building_count_upper_tail_p995", ledger["exclude_graph_scale_p995"], True),
        ("S06", "large_sparse_combination", ledger["S06"], False),
    ]
    removed = pd.Series(False, index=ledger.index)
    rows = []
    for rule_id, reason, mask, triggers_exclusion in rules:
        mask = _bool_series(mask)
        newly = mask & ~removed if triggers_exclusion else pd.Series(False, index=ledger.index)
        if triggers_exclusion:
            removed |= mask
        rows.append(
            {
                "rule_id": rule_id,
                "reason": reason,
                "input_count": input_count,
                "independent_hit_count": int(mask.sum()),
                "newly_excluded_count": int(newly.sum()),
                "cumulative_excluded_count": int(removed.sum()),
                "remaining_count": int(input_count - removed.sum()),
            }
        )
    return pd.DataFrame(rows)


def _summary_reasons(ledger: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for rule in HARD_RULES:
        rows.append((rule, rule, int(ledger[rule].sum()), True))
    rows.append(("H01-H05", "any_hard_rule", int(ledger["hard_flag_count"].gt(0).sum()), True))
    soft_names = {
        "S01": "area_upper_tail_p99",
        "S02": "elongation_lower_tail_p01_retain",
        "S03": "compactness_lower_tail_p01",
        "S04": "building_count_upper_tail_p99_retain",
        "S05": "bcr_lower_tail_p01",
        "S06": "large_sparse_combination",
    }
    for rule in SOFT_RULES:
        rows.append((rule, soft_names[rule], int(ledger[rule].sum()), rule in {"S01", "S03", "S05"}))
    rows.append(("P995", "building_count_upper_tail_p995", int(ledger["exclude_graph_scale_p995"].sum()), True))
    return pd.DataFrame(rows, columns=["rule_id", "reason", "independent_hit_count", "triggers_exclusion"])


def _summary_overlap(ledger: pd.DataFrame) -> pd.DataFrame:
    excluded = ledger.loc[ledger["final_decision"].eq("exclude")]
    overlap = (
        excluded.groupby("final_exclusion_reasons", dropna=False, sort=True)
        .size()
        .rename("block_count")
        .reset_index()
    )
    overlap["reason_count"] = overlap["final_exclusion_reasons"].map(
        lambda value: len(_split_reasons(value))
    )
    return overlap[["final_exclusion_reasons", "reason_count", "block_count"]]


def _summary_city(ledger: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for city, group in ledger.groupby("city", sort=True, dropna=False):
        eligible = group["final_decision"].eq("eligible")
        rows.append(
            {
                "city": city,
                "input_count": len(group),
                "excluded_count": int((~eligible).sum()),
                "eligible_count": int(eligible.sum()),
                "eligible_ratio": float(eligible.mean()),
                "eligible_building_nodes": int(group.loc[eligible, "lod1_building_count"].sum()),
                "eligible_S02_count": int((eligible & group["S02"]).sum()),
                "eligible_original_S04_count": int((eligible & group["S04"]).sum()),
            }
        )
    return pd.DataFrame(rows)


def _summary_graph_scale(ledger: pd.DataFrame) -> pd.DataFrame:
    cohorts = {
        "input_all": pd.Series(True, index=ledger.index),
        "hard_eligible": ledger["hard_flag_count"].eq(0),
        "graph_scale_excluded_p995": ledger["exclude_graph_scale_p995"],
        "final_eligible": ledger["graph_encoding_eligible"],
    }
    rows = []
    for cohort, mask in cohorts.items():
        values = ledger.loc[mask, "lod1_building_count"]
        rows.append(
            {
                "cohort": cohort,
                "block_count": int(len(values)),
                "building_node_count": int(values.sum()),
                "min_building_count": float(values.min()),
                "median_building_count": float(values.quantile(0.5)),
                "p95_building_count": float(values.quantile(0.95)),
                "p99_building_count": float(values.quantile(0.99)),
                "p995_building_count": float(values.quantile(0.995)),
                "max_building_count": float(values.max()),
            }
        )
    return pd.DataFrame(rows)


def build_final_summaries(ledger: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build all compact summaries from the same final-decision ledger."""
    return {
        "final_filter_flow": _summary_flow(ledger),
        "final_filter_reason_summary": _summary_reasons(ledger),
        "final_filter_overlap_summary": _summary_overlap(ledger),
        "final_summary_by_city": _summary_city(ledger),
        "graph_scale_summary": _summary_graph_scale(ledger),
    }


def _load_rules() -> dict[str, Any]:
    return json.loads((PHASE_DIR / "final_screening_rules_v0.2.json").read_text(encoding="utf-8"))


def _validate_run_id(value: str, label: str) -> None:
    if not RUN_ID_PATTERN.fullmatch(value):
        raise ValueError(f"{label} contains unsupported characters: {value!r}")


def _result_paths(results_dir: Path, run_id: str) -> dict[str, Path]:
    bulk_dir = results_dir / "bulk" / run_id
    audit_dir = results_dir / "audit" / run_id
    return {
        "bulk_dir": bulk_dir,
        "audit_dir": audit_dir,
        "ledger": bulk_dir / "final_screening_ledger.csv.gz",
        "excluded": bulk_dir / "excluded_blocks.csv.gz",
        "manifest": results_dir / "dataset" / "graph_encoding_manifest.csv.gz",
        "input_snapshot": results_dir / "input_snapshot.json",
        "freeze": results_dir / "freeze.json",
        "environment": results_dir / "environment.json",
    }


def _frozen_metrics(ledger: pd.DataFrame, threshold: float) -> dict[str, Any]:
    eligible = ledger["graph_encoding_eligible"]
    return {
        "input_rows": int(len(ledger)),
        "input_cities": int(ledger["city"].nunique(dropna=True)),
        "hard_eligible_rows": int(ledger["hard_flag_count"].eq(0).sum()),
        "building_count_p995_threshold": float(threshold),
        "final_excluded": int((~eligible).sum()),
        "final_eligible": int(eligible.sum()),
        "final_eligible_ratio": float(eligible.mean()),
        "final_eligible_building_nodes": int(ledger.loc[eligible, "lod1_building_count"].sum()),
        "hard_any_hits": int(ledger["hard_flag_count"].gt(0).sum()),
        "S01_hits": int(ledger["S01"].sum()),
        "S03_hits": int(ledger["S03"].sum()),
        "S05_hits": int(ledger["S05"].sum()),
        "graph_scale_p995_hits": int(ledger["exclude_graph_scale_p995"].sum()),
        "eligible_original_S04": int((eligible & ledger["S04"]).sum()),
        "eligible_S02": int((eligible & ledger["S02"]).sum()),
        "eligible_cities": int(ledger.loc[eligible, "city"].nunique(dropna=True)),
    }


def _assert_expected(metrics: dict[str, Any], rules: dict[str, Any]) -> None:
    expected = {
        "input_rows": rules["expected_input_rows"],
        "input_cities": rules["expected_input_cities"],
        "hard_eligible_rows": rules["expected_hard_eligible_rows"],
        "building_count_p995_threshold": rules["expected_building_count_p995_threshold"],
        **rules["expected_frozen_counts"],
    }
    mismatches = [
        f"{key}: expected {value}, got {metrics.get(key)}"
        for key, value in expected.items()
        if metrics.get(key) != value
    ]
    if mismatches:
        raise ValueError("Frozen-count validation failed: " + "; ".join(mismatches))


def _write_evidence_pack(
    path: Path,
    metrics: dict[str, Any],
    paths: dict[str, Path],
    verification: dict[str, Any] | None = None,
) -> None:
    def display(artifact: Path) -> str:
        try:
            return artifact.relative_to(path.parent).as_posix()
        except ValueError:
            return str(artifact)

    issues = [] if verification is None else verification.get("issues", [])
    verification_status = "PENDING" if verification is None else verification.get("status", "UNKNOWN")
    lines = [
                "# Phase 3.5B final sample freeze evidence pack",
                "",
                f"- Input blocks: {metrics['input_rows']:,}",
                f"- Final excluded: {metrics['final_excluded']:,}",
                f"- Graph-encoding eligible: {metrics['final_eligible']:,}",
                f"- Eligible building nodes: {metrics['final_eligible_building_nodes']:,}",
                f"- P99.5 threshold: {metrics['building_count_p995_threshold']:g}",
                f"- Eligible cities: {metrics['eligible_cities']}",
                "",
                "Generated artifacts:",
                f"- `{display(paths['ledger'])}`",
                f"- `{display(paths['excluded'])}`",
                f"- `{display(paths['manifest'])}`",
                "",
                "Artifact SHA256:",
                f"- final_screening_ledger: `{sha256_file(paths['ledger'])}`",
                f"- excluded_blocks: `{sha256_file(paths['excluded'])}`",
                f"- graph_encoding_manifest: `{sha256_file(paths['manifest'])}`",
                "",
                f"Verification status: **{verification_status}**",
                f"Failed checks: {len(issues)}",
            ]
    lines.extend(f"- {item['check']}: {item['detail']}" for item in issues)
    lines.extend(
        [
            "",
            "未访问数据库、未构图。",
            "This phase freezes the graph-encoding entry list; it does not build graphs.",
            "",
        ]
    )
    path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def freeze_run(
    source_path: Path,
    results_dir: Path,
    source_run_id: str = DEFAULT_SOURCE_RUN_ID,
    run_id: str = DEFAULT_RUN_ID,
    enforce_expected: bool = True,
) -> dict[str, Any]:
    """Freeze one source ledger into final ledger, exclusions, manifest and audits."""
    source_path = Path(source_path).resolve()
    results_dir = Path(results_dir).resolve()
    _validate_run_id(source_run_id, "source_run_id")
    _validate_run_id(run_id, "run_id")
    if not source_path.is_file():
        raise FileNotFoundError(source_path)

    paths = _result_paths(results_dir, run_id)
    collisions = [path for path in paths.values() if path.exists()]
    if collisions:
        joined = ", ".join(str(path) for path in collisions)
        raise FileExistsError(f"Refusing to overwrite existing Phase 3.5B outputs: {joined}")

    source = pd.read_csv(
        source_path,
        encoding="utf-8-sig",
        float_precision="round_trip",
        low_memory=False,
    )
    _validate_rule_input(source)
    if "rule_version" in source and not source["rule_version"].astype("string").eq("0.1").all():
        raise ValueError("Phase 3.5B requires Phase 3.5A rule_version 0.1 only")
    threshold = compute_p995_threshold(source)
    final_ledger = apply_final_rules(source, threshold)
    final_ledger["source_run_id"] = source_run_id
    manifest = build_graph_manifest(final_ledger)
    excluded = final_ledger.loc[final_ledger["final_decision"].eq("exclude")].reset_index(drop=True)
    metrics = _frozen_metrics(final_ledger, threshold)
    rules = _load_rules()
    if enforce_expected:
        expected_hash = rules.get("expected_source_sha256")
        actual_hash = sha256_file(source_path)
        if expected_hash and actual_hash != expected_hash:
            raise ValueError(f"Unexpected Phase 3.5A source SHA256: {actual_hash}")
        _assert_expected(metrics, rules)

    write_gzip_csv(paths["ledger"], final_ledger)
    write_gzip_csv(paths["excluded"], excluded)
    write_gzip_csv(paths["manifest"], manifest)
    summaries = build_final_summaries(final_ledger)
    for name, frame in summaries.items():
        write_csv(paths["audit_dir"] / f"{name}.csv", frame)

    generated_at = datetime.now(timezone.utc).isoformat()
    snapshot = {
        "source_phase": rules["source_phase"],
        "source_run_id": source_run_id,
        "source_path": str(source_path),
        "source_sha256": sha256_file(source_path),
        "source_size_bytes": source_path.stat().st_size,
        "source_rows": metrics["input_rows"],
        "source_cities": metrics["input_cities"],
        "source_unique_city_block_keys": int(
            source[["city", "block_id"]].drop_duplicates().shape[0]
        ),
        "source_key_is_unique": bool(~source[["city", "block_id"]].duplicated().any()),
        "source_rule_versions": sorted(source["rule_version"].astype("string").unique().tolist()),
        "captured_at_utc": generated_at,
    }
    freeze = {
        "experiment_id": rules["experiment_id"],
        "status": "FROZEN",
        "final_rule_version": FINAL_RULE_VERSION,
        "source_run_id": source_run_id,
        "run_id": run_id,
        **metrics,
        "artifact_sha256": {
            "final_screening_ledger": sha256_file(paths["ledger"]),
            "excluded_blocks": sha256_file(paths["excluded"]),
            "graph_encoding_manifest": sha256_file(paths["manifest"]),
        },
        "generated_at_utc": generated_at,
        "graph_construction_started": False,
    }
    environment = {
        "python": sys.version.split()[0],
        "pandas": pd.__version__,
        "platform": platform.platform(),
        "database_accessed": False,
        "generated_at_utc": generated_at,
    }
    write_json(paths["input_snapshot"], snapshot)
    write_json(paths["freeze"], freeze)
    write_json(paths["environment"], environment)
    _write_evidence_pack(results_dir / "phase3_5b_evidence_pack.md", metrics, paths)
    (results_dir / "run_log.txt").write_text(
        f"{generated_at} froze {metrics['final_eligible']} eligible blocks as {run_id}\n",
        encoding="utf-8",
    )
    return freeze


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run-id", default=DEFAULT_SOURCE_RUN_ID)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    args = parser.parse_args()
    source_path = (
        PHASE_DIR.parent
        / "phase3_5a_block_screening_v01"
        / "results"
        / "bulk"
        / args.source_run_id
        / "block_screening_ledger.csv.gz"
    )
    result = freeze_run(
        source_path=source_path,
        results_dir=RESULTS_DIR,
        source_run_id=args.source_run_id,
        run_id=args.run_id,
        enforce_expected=True,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
