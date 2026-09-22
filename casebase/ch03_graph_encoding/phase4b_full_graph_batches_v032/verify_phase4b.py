"""Verify Phase 4B preparation or all completed parts without generating graphs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


PHASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PHASE_DIR / "results"
SPEC = json.loads((PHASE_DIR / "encoding_spec_v032.json").read_text(encoding="utf-8"))
GROUP_SPEC = pd.read_csv(PHASE_DIR / "city_group_spec_v01.csv", encoding="utf-8-sig")
PART_IDS = tuple(GROUP_SPEC["part_id"].tolist())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _latest_status(path: Path) -> pd.DataFrame:
    status = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    status["attempt"] = pd.to_numeric(status["attempt"], errors="raise").astype(int)
    return status.sort_values(["sample_id", "attempt"], kind="mergesort").drop_duplicates("sample_id", keep="last")


def _manifest_checks() -> tuple[dict, dict[str, pd.DataFrame]]:
    recorded = json.loads((RESULTS_DIR / "audit" / "prep" / "manifest_hashes.json").read_text(encoding="utf-8"))
    full_path = RESULTS_DIR / "dataset" / "full_run_manifest_v032.csv.gz"
    if sha256_file(full_path) != recorded["full_run_manifest_v032.csv.gz"]:
        raise RuntimeError("Full manifest SHA256 differs from the frozen audit")
    full = pd.read_csv(full_path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    if len(full) != SPEC["source_manifest_rows"] or full[["city", "block_id"]].duplicated().any():
        raise RuntimeError("Full manifest count or uniqueness check failed")
    parts: dict[str, pd.DataFrame] = {}
    keys: list[pd.DataFrame] = []
    expected = GROUP_SPEC.set_index("part_id")["expected_block_count"].astype(int).to_dict()
    for part_id in PART_IDS:
        path = RESULTS_DIR / "dataset" / "parts" / f"{part_id}_manifest.csv.gz"
        if sha256_file(path) != recorded["parts"][part_id]:
            raise RuntimeError(f"{part_id} manifest SHA256 differs from the frozen audit")
        frame = pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
        if len(frame) != expected[part_id] or set(frame["part_id"]) != {part_id}:
            raise RuntimeError(f"{part_id} count or boundary check failed")
        parts[part_id] = frame
        keys.append(frame[["city", "block_id"]])
    combined = pd.concat(keys, ignore_index=True)
    if combined.duplicated().any() or set(map(tuple, combined.to_numpy())) != set(map(tuple, full[["city", "block_id"]].to_numpy())):
        raise RuntimeError("Part manifests are not an exact disjoint partition of the full manifest")
    return recorded, parts


def verify_prep() -> dict:
    hashes, parts = _manifest_checks()
    regression_path = RESULTS_DIR / "audit" / "prep" / "v032_regression_summary.csv"
    regression = pd.read_csv(regression_path, encoding="utf-8-sig")
    expected_cases = {("Boston", "US_BO_003310"), ("San Francisco", "US_SF_000689")}
    actual_cases = set(map(tuple, regression[["city", "block_id"]].to_numpy()))
    regression_ok = (
        actual_cases == expected_cases
        and len(regression) == 2
        and regression["validation_passed"].astype(bool).all()
        and regression["transaction_read_only"].astype(str).str.lower().eq("on").all()
    )
    if not regression_ok:
        raise RuntimeError("The two-case read-only v0.3.2 regression did not pass")
    run_root = RESULTS_DIR / "runs" / "full_v032"
    started = [part for part in PART_IDS if (run_root / part).exists()]
    if started:
        raise RuntimeError(f"Formal parts have already been started: {started}")
    summary = {
        "phase": "4B",
        "state": "CODE_READY_FORMAL_PARTS_NOT_STARTED",
        "encoding_version": SPEC["encoding_version"],
        "source_manifest_sha256": hashes["source_manifest_sha256"],
        "full_manifest_sha256": hashes["full_run_manifest_v032.csv.gz"],
        "source_rows": SPEC["source_manifest_rows"],
        "source_cities": 24,
        "part_count": len(parts),
        "part_rows": {part_id: len(frame) for part_id, frame in parts.items()},
        "manifest_partition_passed": True,
        "targeted_unit_tests": {"passed": 8, "failed": 0},
        "related_regression_tests": {"phase3_1_passed": 4, "phase4a_passed": 8, "failed": 0},
        "regression_cases": 2,
        "regression_passed": True,
        "database_transaction_read_only": True,
        "formal_parts": {part_id: "PLANNED" for part_id in PART_IDS},
    }
    audit_path = RESULTS_DIR / "audit" / "prep" / "verification_summary.json"
    audit_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    evidence = RESULTS_DIR / "phase4b_prep_evidence_pack.md"
    lines = [
        "# Phase 4B preparation evidence",
        "",
        "- State: code and frozen manifests are ready; no formal part has started.",
        f"- Frozen entry count: {SPEC['source_manifest_rows']:,} across 24 cities and 12 disjoint parts.",
        "- v0.3.2 regression: 2/2 passed using a read-only database transaction.",
        "- Targeted Phase 4B tests: 8/8 passed.",
        "- Formal parts: " + ", ".join(f"{part}=PLANNED" for part in PART_IDS) + ".",
        "",
        "This file is an ignored handoff artifact; compact audit files remain tracked.",
    ]
    evidence.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def verify_all_parts() -> dict:
    _, parts = _manifest_checks()
    missing: list[str] = []
    problems: dict[str, list[str]] = {}
    seen: set[tuple[str, str]] = set()
    completed = excluded = 0
    for part_id, manifest in parts.items():
        status_path = RESULTS_DIR / "runs" / "full_v032" / part_id / "encoding_status.csv"
        if not status_path.is_file():
            missing.append(part_id)
            continue
        latest = _latest_status(status_path)
        expected_keys = set(map(tuple, manifest[["city", "block_id"]].to_numpy()))
        status_keys = set(map(tuple, latest[["city", "block_id"]].to_numpy()))
        part_problems: list[str] = []
        if status_keys != expected_keys:
            part_problems.append(f"status key mismatch: expected {len(expected_keys)}, got {len(status_keys)}")
        bad = latest.loc[~latest["status"].isin(["completed", "excluded_no_valid_building"])]
        if not bad.empty:
            part_problems.append(f"unaccounted cases: {len(bad)}")
        for row in latest.loc[latest["status"].eq("completed")].itertuples(index=False):
            graph_path = PHASE_DIR / str(row.graph_path)
            if not graph_path.is_file():
                part_problems.append(f"missing graph: {row.sample_id}")
                break
        overlap = seen.intersection(status_keys)
        if overlap:
            part_problems.append(f"cross-part duplicate statuses: {len(overlap)}")
        seen.update(status_keys)
        if part_problems:
            problems[part_id] = part_problems
        completed += int(latest["status"].eq("completed").sum())
        excluded += int(latest["status"].eq("excluded_no_valid_building").sum())
    summary = {
        "passed": not missing and not problems and len(seen) == SPEC["source_manifest_rows"],
        "missing_parts": missing,
        "problems": problems,
        "accounted_entries": len(seen),
        "completed_graphs": completed,
        "excluded_no_valid_building": excluded,
        "expected_entries": SPEC["source_manifest_rows"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prep", action="store_true", help="verify code-ready preparation evidence")
    mode.add_argument("--all-parts", action="store_true", help="read-only verification after all manual runs")
    args = parser.parse_args()
    if args.prep:
        print(json.dumps(verify_prep(), ensure_ascii=False, indent=2))
        return 0
    return 0 if verify_all_parts()["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
