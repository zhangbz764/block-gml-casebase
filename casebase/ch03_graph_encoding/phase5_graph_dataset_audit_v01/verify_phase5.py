"""Automated verification for a completed Phase 5 audit run."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

from build_graph_index import phase5_code_bundle_sha256
from phase4b_gate import validate_phase4b_gate
from registry import load_audit_spec, load_registry, sha256_file


PHASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PHASE_DIR / "results"
PHASE4B_DIR = PHASE_DIR.parent / "phase4b_full_graph_batches_v032"


def validate_part_audit_meta(meta: dict, current_phase5_bundle: str, part_id: str) -> list[str]:
    issues: list[str] = []
    if not meta.get("passed"):
        issues.append(f"{part_id} audit shard did not pass")
    if meta.get("phase5_code_bundle_sha256") != current_phase5_bundle:
        issues.append(f"{part_id} Phase 5 code bundle differs from the current audit code")
    return issues


def manual_review_status(review: pd.DataFrame, manual: pd.DataFrame, expected_count: int) -> tuple[str, list[str]]:
    issues: list[str] = []
    expected_ids = review.get("sample_id", pd.Series(dtype=str)).astype(str).tolist()
    manual_ids = manual.get("sample_id", pd.Series(dtype=str)).astype(str).tolist()
    if len(manual) != expected_count or len(review) != expected_count:
        issues.append(f"Manual review row count does not match the {expected_count}-case review manifest")
    if set(manual_ids) != set(expected_ids):
        issues.append("Manual review sample ID set differs from the review manifest")
    if manual_ids != expected_ids:
        issues.append("Manual review sample ID order differs from the review manifest")
    if issues:
        return "MANIFEST_MISMATCH", issues
    if "visual_status" not in manual or not manual["visual_status"].astype(str).str.strip().ne("").all():
        return "PENDING", []
    return "COMPLETED", []


def validate_phase5_tables(index: pd.DataFrame, features: pd.DataFrame, review: pd.DataFrame, spec: dict) -> list[str]:
    issues: list[str] = []
    expected = int(spec["expected_entries"])
    if len(index) != expected:
        issues.append(f"Graph index has {len(index)} rows, expected {expected}")
    if index.get("sample_id", pd.Series(dtype=str)).astype(str).duplicated().any():
        issues.append("Graph index has duplicate sample_id")
    if index.get("block_id", pd.Series(dtype=str)).size and index[["city", "block_id"]].astype(str).duplicated().any():
        issues.append("Graph index has duplicate city + block_id")
    bad_status = index.loc[~index["case_status"].isin(["completed", "excluded_no_valid_building"])]
    if not bad_status.empty:
        issues.append(f"Graph index has {len(bad_status)} failed/interrupted/missing entries")
    completed = index.loc[index["case_status"].eq("completed")]
    for field in ("graph_exists", "graph_readable", "validation_passed", "audit_passed"):
        if field not in completed or not completed[field].astype(str).str.lower().isin(["true", "1"]).all():
            issues.append(f"Completed Graphs do not all satisfy {field}")
    cities = set(index["city"].astype(str))
    expected_city_names = set(map(str, spec.get("expected_city_names", [])))
    if len(cities) != int(spec["expected_cities"]):
        issues.append(f"Graph index has {len(cities)} cities, expected {spec['expected_cities']}")
    if expected_city_names and cities != expected_city_names:
        issues.append("Graph index city set differs from the frozen manifest")
    global_features = features.loc[features["scope"].eq("__global__")]
    if len(global_features) != 103 or global_features["field"].nunique() != 103:
        issues.append("Global feature completeness table is not the frozen 103 fields")
    entity_counts = global_features.groupby("entity").size().to_dict()
    role_counts = global_features.groupby("role").size().to_dict()
    if entity_counts != spec["registry_entity_counts"]:
        issues.append("Feature entity counts differ from the registry")
    if role_counts != spec["registry_role_counts"]:
        issues.append("Feature role counts differ from the registry")
    city_features = features.loc[~features["scope"].eq("__global__")]
    if len(city_features) != 103 * int(spec["expected_cities"]):
        issues.append("City feature completeness table does not contain 103 rows per city")
    if len(review) != int(spec["review_case_count"]):
        issues.append(f"Review manifest has {len(review)} rows, expected {spec['review_case_count']}")
    if not review.empty and not review.groupby("city").size().eq(int(spec["review_cases_per_city"])).all():
        issues.append("Review manifest does not contain two cases per city")
    if not review.empty and review["sample_id"].astype(str).duplicated().any():
        issues.append("Review manifest has duplicate samples")
    text_columns = index.select_dtypes(include="object")
    if text_columns.apply(lambda column: column.astype(str).str.contains("pilot", case=False, na=False).any()).any():
        issues.append("Formal index contains pilot references")
    return issues


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def verify_automated(results_root: Path = RESULTS_DIR, spec: dict | None = None) -> dict:
    spec = dict(spec or load_audit_spec(PHASE_DIR / "audit_spec_v01.json"))
    full_manifest = PHASE4B_DIR / "results" / "dataset" / "full_run_manifest_v032.csv.gz"
    if "expected_city_names" not in spec and full_manifest.is_file():
        frozen = pd.read_csv(full_manifest, encoding="utf-8-sig", usecols=["city"], dtype=str)
        spec["expected_city_names"] = sorted(frozen["city"].dropna().unique())
    full_name = spec.get("formal_output_name", "full_v01")
    audit_dir = results_root / "audit" / full_name
    dataset_dir = results_root / "dataset"
    review_dir = results_root / "review" / full_name
    gate_path = audit_dir / "phase4b_all_parts_verification.json"
    if not gate_path.is_file():
        gate_issues = ["Phase 4B verification capture is missing"]
    else:
        gate_issues = validate_phase4b_gate(json.loads(gate_path.read_text(encoding="utf-8")), spec)
    issues = list(gate_issues)
    current_phase5_bundle = phase5_code_bundle_sha256()
    input_snapshot_path = results_root / "input_snapshot.json"
    if not input_snapshot_path.is_file():
        issues.append("Phase 5 input_snapshot.json is missing")
    else:
        input_snapshot = json.loads(input_snapshot_path.read_text(encoding="utf-8"))
        if input_snapshot.get("phase5_code_bundle_sha256") != current_phase5_bundle:
            issues.append("Phase 5 input snapshot uses a different code bundle")
    part_ids = list(spec["part_ids"])
    for part_id in part_ids:
        paths = {
            "index": dataset_dir / "index_parts" / f"{part_id}_graph_index.csv.gz",
            "features": audit_dir / "parts" / f"{part_id}_feature_counts.csv",
            "issues": audit_dir / "parts" / f"{part_id}_issues.csv",
        }
        meta_path = audit_dir / "parts" / f"{part_id}_audit_meta.json"
        if not meta_path.is_file() or not all(path.is_file() for path in paths.values()):
            issues.append(f"{part_id} audit shard is missing")
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        issues.extend(validate_part_audit_meta(meta, current_phase5_bundle, part_id))
        for name, path in paths.items():
            if sha256_file(path) != meta.get("output_sha256", {}).get(name):
                issues.append(f"{part_id} {name} output SHA256 mismatch")
    required = {
        "index": dataset_dir / "graph_dataset_index.csv.gz",
        "global_features": audit_dir / "feature_completeness_global.csv",
        "city_features": audit_dir / "feature_completeness_by_city.csv",
        "review": review_dir / "review_manifest.csv",
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    review = pd.DataFrame()
    if missing:
        issues.append(f"Required Phase 5 outputs are missing: {missing}")
    else:
        index = pd.read_csv(required["index"], encoding="utf-8-sig", low_memory=False)
        global_features = pd.read_csv(required["global_features"], encoding="utf-8-sig")
        city_features = pd.read_csv(required["city_features"], encoding="utf-8-sig").rename(columns={"city": "scope"})
        features = pd.concat([global_features.assign(scope="__global__"), city_features], ignore_index=True)
        review = pd.read_csv(required["review"], encoding="utf-8-sig")
        issues.extend(validate_phase5_tables(index, features, review, spec))
    manual_path = review_dir / "manual_review.csv"
    manual_status = "PENDING"
    if manual_path.is_file():
        manual = pd.read_csv(manual_path, encoding="utf-8-sig", keep_default_na=False)
        manual_status, manual_issues = manual_review_status(review, manual, int(spec["review_case_count"]))
        issues.extend(manual_issues)
    automated = not issues
    state = "ANALYZED_AUTOMATED_MANUAL_REVIEW_PENDING" if automated and manual_status == "PENDING" else (
        "ANALYZED_AUTOMATED_AND_MANUAL_REVIEWED" if automated else "AUDIT_FAILED"
    )
    summary = {"automated_audit_passed": automated, "manual_visual_review_status": manual_status,
               "evidence_state": state, "issues": issues}
    _atomic_json(audit_dir / "verification_summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--automated", action="store_true", required=True)
    args = parser.parse_args()
    if not args.automated:
        return 2
    result = verify_automated()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["automated_audit_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
