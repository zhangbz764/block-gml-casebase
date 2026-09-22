"""Independently verify Phase 3.5B artifacts by reading every result from disk."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from freeze_final_sample import (
    DEFAULT_RUN_ID,
    DEFAULT_SOURCE_RUN_ID,
    FINAL_RULE_VERSION,
    RESULTS_DIR,
    _assert_expected,
    _frozen_metrics,
    _load_rules,
    _result_paths,
    _write_evidence_pack,
    apply_final_rules,
    build_final_summaries,
    build_graph_manifest,
    sha256_file,
    stable_csv_bytes,
    write_json,
)


FINAL_COLUMNS = [
    "final_rule_version",
    "building_count_p995_threshold",
    "exclude_graph_scale_p995",
    "final_decision",
    "graph_encoding_eligible",
    "primary_final_reason",
    "final_exclusion_reasons",
]


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        encoding="utf-8-sig",
        float_precision="round_trip",
        low_memory=False,
    )


def _same_frame(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    return stable_csv_bytes(left.reset_index(drop=True)) == stable_csv_bytes(right.reset_index(drop=True))


def verify_run(
    results_dir: Path,
    run_id: str = DEFAULT_RUN_ID,
    source_run_id: str = DEFAULT_SOURCE_RUN_ID,
    enforce_expected: bool = True,
) -> dict[str, Any]:
    """Read frozen files afresh, recompute all decisions, and save a verification record."""
    results_dir = Path(results_dir).resolve()
    paths = _result_paths(results_dir, run_id)
    issues: list[dict[str, str]] = []

    def add(check: str, detail: str) -> None:
        issues.append({"check": check, "detail": detail})

    required = [
        paths["ledger"],
        paths["excluded"],
        paths["manifest"],
        paths["input_snapshot"],
        paths["freeze"],
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        add("required_files", "missing: " + ", ".join(missing))
        result = {
            "run_id": run_id,
            "source_run_id": source_run_id,
            "status": "FAILED",
            "passed": False,
            "issue_count": len(issues),
            "issues": issues,
        }
        paths["audit_dir"].mkdir(parents=True, exist_ok=True)
        write_json(paths["audit_dir"] / "verification_summary.json", result)
        return result

    ledger = _read_csv(paths["ledger"])
    excluded = _read_csv(paths["excluded"])
    manifest = _read_csv(paths["manifest"])
    freeze = json.loads(paths["freeze"].read_text(encoding="utf-8"))
    snapshot = json.loads(paths["input_snapshot"].read_text(encoding="utf-8"))
    rules = _load_rules()

    threshold_values = pd.to_numeric(
        ledger["building_count_p995_threshold"], errors="coerce"
    ).dropna().unique()
    if len(threshold_values) != 1:
        add("threshold_domain", f"expected one threshold, got {threshold_values.tolist()}")
        threshold = float("nan")
    else:
        threshold = float(threshold_values[0])

    if ledger[["city", "block_id"]].duplicated().any():
        add("ledger_unique_key", "duplicate city + block_id exists")
    if set(ledger["final_decision"].dropna().unique()) - {"exclude", "eligible"}:
        add("decision_domain", "final_decision contains an unsupported value")
    if not ledger["final_rule_version"].astype("string").eq(FINAL_RULE_VERSION).all():
        add("rule_version", "a row does not use final rule version 0.2")
    if not ledger["source_run_id"].astype("string").eq(source_run_id).all():
        add("source_run_id", "a row has an unexpected source_run_id")

    if len(threshold_values) == 1:
        recomputed = apply_final_rules(ledger, threshold)
        if not _same_frame(ledger[FINAL_COLUMNS], recomputed[FINAL_COLUMNS]):
            add("decision_recompute", "saved final decisions differ from a fresh rule recomputation")

    expected_excluded = ledger.loc[ledger["final_decision"].eq("exclude")].reset_index(drop=True)
    if not _same_frame(excluded, expected_excluded):
        add("excluded_ledger", "excluded_blocks is not the exact excluded subset")
    expected_manifest = build_graph_manifest(ledger)
    if not _same_frame(manifest, expected_manifest):
        add("graph_manifest", "graph manifest is not the exact eligible subset")
    if manifest[["city", "block_id"]].duplicated().any():
        add("manifest_unique_key", "duplicate city + block_id exists in graph manifest")
    if not manifest["graph_encoding_eligible"].astype(bool).all():
        add("manifest_eligibility", "graph manifest contains an ineligible block")
    if ledger.loc[ledger["graph_encoding_eligible"].astype(bool), "lod1_building_count"].ge(threshold).any():
        add("eligible_graph_scale", "eligible block reaches or exceeds the frozen P99.5 threshold")

    expected_summaries = build_final_summaries(ledger)
    for name, expected in expected_summaries.items():
        saved_path = paths["audit_dir"] / f"{name}.csv"
        if not saved_path.is_file():
            add("audit_summary", f"missing {saved_path.name}")
            continue
        saved = _read_csv(saved_path)
        if not _same_frame(saved, expected):
            add("audit_summary", f"{saved_path.name} differs from fresh recomputation")

    metrics = _frozen_metrics(ledger, threshold)
    if enforce_expected:
        try:
            _assert_expected(metrics, rules)
        except ValueError as exc:
            add("frozen_counts", str(exc))
    if snapshot.get("source_run_id") != source_run_id:
        add("snapshot_source_run", "input snapshot source_run_id differs")
    snapshot_source = Path(snapshot.get("source_path", ""))
    if snapshot_source.is_file() and sha256_file(snapshot_source) != snapshot.get("source_sha256"):
        add("snapshot_source_hash", "source ledger hash differs from input snapshot")

    actual_hashes = {
        "final_screening_ledger": sha256_file(paths["ledger"]),
        "excluded_blocks": sha256_file(paths["excluded"]),
        "graph_encoding_manifest": sha256_file(paths["manifest"]),
    }
    if freeze.get("artifact_sha256") != actual_hashes:
        add("artifact_hashes", "one or more frozen artifact hashes differ")

    result = {
        "experiment_id": rules["experiment_id"],
        "final_rule_version": FINAL_RULE_VERSION,
        "run_id": run_id,
        "source_run_id": source_run_id,
        "status": "VERIFIED" if not issues else "FAILED",
        "passed": not issues,
        **metrics,
        "artifact_sha256": actual_hashes,
        "issue_count": len(issues),
        "issues": issues,
        "database_accessed": False,
        "graph_construction_started": False,
    }
    write_json(paths["audit_dir"] / "verification_summary.json", result)
    _write_evidence_pack(
        results_dir / "phase3_5b_evidence_pack.md",
        metrics,
        paths,
        verification=result,
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run-id", default=DEFAULT_SOURCE_RUN_ID)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    args = parser.parse_args()
    result = verify_run(
        results_dir=RESULTS_DIR,
        run_id=args.run_id,
        source_run_id=args.source_run_id,
        enforce_expected=True,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
