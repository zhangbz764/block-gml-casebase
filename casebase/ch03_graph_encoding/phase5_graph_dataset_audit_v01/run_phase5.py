"""Explicit Phase 5 runner. There is deliberately no default formal mode."""

from __future__ import annotations

import argparse
import gzip
import json
import os
from pathlib import Path
import platform
import sys
from datetime import datetime, timezone

import matplotlib
import pandas as pd
import shapely

from build_graph_index import build_part_index, _read_graph, phase5_code_bundle_sha256
from merge_graph_audit import merge_feature_counts, merge_part_indexes, write_dataset_summaries
from phase4b_gate import validate_phase4b_gate
from registry import load_audit_spec, sha256_file
from render_visual_review import build_review_html, render_review_case, write_manual_review
from select_review_sample import select_review_cases
from verify_phase5 import verify_automated
from write_researcher_summary import write_researcher_summary


PHASE_DIR = Path(__file__).resolve().parent
PHASE4B_DIR = PHASE_DIR.parent / "phase4b_full_graph_batches_v032"
RESULTS_DIR = PHASE_DIR / "results"


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _write_once_or_match(path: Path, value: dict) -> None:
    if path.is_file():
        if json.loads(path.read_text(encoding="utf-8")) != value:
            raise RuntimeError(f"Existing frozen preparation file differs: {path}")
        return
    _atomic_json(path, value)


def _prepare_snapshots(spec: dict, gate_path: Path) -> None:
    parts = {}
    for part_id in spec["part_ids"]:
        manifest = PHASE4B_DIR / "results/dataset/parts" / f"{part_id}_manifest.csv.gz"
        status = PHASE4B_DIR / "results/runs/full_v032" / part_id / "encoding_status.csv"
        parts[part_id] = {"manifest_sha256": sha256_file(manifest), "status_ledger_sha256": sha256_file(status)}
    snapshot = {"phase4b_verification_sha256": sha256_file(gate_path),
                "phase5_code_bundle_sha256": spec["phase5_code_bundle_sha256"], "parts": parts}
    environment = {"python": platform.python_version(), "pandas": pd.__version__, "shapely": shapely.__version__, "matplotlib": matplotlib.__version__}
    _write_once_or_match(RESULTS_DIR / "input_snapshot.json", snapshot)
    _write_once_or_match(RESULTS_DIR / "environment.json", environment)


def _write_full_index(index: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    index.to_csv(temporary, index=False, encoding="utf-8-sig", lineterminator="\n", compression={"method": "gzip", "mtime": 0})
    os.replace(temporary, path)


def ensure_manual_review_matches(manifest: pd.DataFrame, path: Path) -> None:
    expected = manifest["sample_id"].astype(str).tolist()
    if path.is_file():
        existing = pd.read_csv(path, encoding="utf-8-sig", keep_default_na=False)
        actual = existing.get("sample_id", pd.Series(dtype=str)).astype(str).tolist()
        if actual != expected:
            raise RuntimeError(
                "Existing manual_review.csv does not match this review manifest; rename it for preservation before rerunning"
            )
        return
    write_manual_review(manifest, path)


def _render_review(review: pd.DataFrame, review_dir: Path) -> pd.DataFrame:
    review = review.copy(); image_paths = []
    manual = review_dir / "manual_review.csv"
    ensure_manual_review_matches(review, manual)
    for row in review.to_dict("records"):
        record = _read_graph((PHASE4B_DIR / str(row["graph_path"])).resolve())
        filename = f"{row['sample_id']}.png"
        render_review_case(record, review_dir / "images" / filename, row)
        image_paths.append(f"images/{filename}")
    review["image_path"] = image_paths
    review.to_csv(review_dir / "review_manifest.csv", index=False, encoding="utf-8-sig", lineterminator="\n")
    build_review_html(review, review_dir / "visual_review.html")
    return review


def run_phase5(part_id: str | None, all_parts: bool, run_id: str) -> dict:
    if run_id != "full_v01":
        raise ValueError("Formal Phase 5 run_id must be full_v01")
    spec = load_audit_spec(PHASE_DIR / "audit_spec_v01.json")
    spec["_spec_path"] = str(PHASE_DIR / "audit_spec_v01.json")
    spec["phase5_code_bundle_sha256"] = phase5_code_bundle_sha256()
    audit_dir = RESULTS_DIR / "audit" / run_id
    gate_path = audit_dir / "phase4b_all_parts_verification.json"
    if not gate_path.is_file():
        raise RuntimeError("Capture the Phase 4B all-parts verification before Phase 5")
    gate_issues = validate_phase4b_gate(json.loads(gate_path.read_text(encoding="utf-8")), spec)
    if gate_issues:
        raise RuntimeError("Phase 4B gate failed: " + "; ".join(gate_issues))
    formal_markers = [
        RESULTS_DIR / "dataset" / "graph_dataset_index.csv.gz",
        audit_dir / "global_summary.json",
        RESULTS_DIR / "review" / run_id / "review_manifest.csv",
    ]
    if any(path.exists() for path in formal_markers) and not (RESULTS_DIR / "input_snapshot.json").is_file():
        raise RuntimeError("Existing formal outputs lack a compatible frozen input_snapshot.json")
    _prepare_snapshots(spec, gate_path)
    targets = list(spec["part_ids"]) if all_parts else [str(part_id)]
    part_results = [build_part_index(target, PHASE4B_DIR, RESULTS_DIR, spec) for target in targets]
    if any(not result.get("passed") for result in part_results):
        return {"passed": False, "parts": part_results}
    if not all_parts:
        return {"passed": True, "parts": part_results, "full_merge": "not_requested"}

    index_paths = [RESULTS_DIR / "dataset/index_parts" / f"{part}_graph_index.csv.gz" for part in spec["part_ids"]]
    feature_paths = [audit_dir / "parts" / f"{part}_feature_counts.csv" for part in spec["part_ids"]]
    index = merge_part_indexes(index_paths, spec["expected_entries"])
    features = merge_feature_counts(feature_paths)
    _write_full_index(index, RESULTS_DIR / "dataset/graph_dataset_index.csv.gz")
    global_summary = write_dataset_summaries(index, features, audit_dir)
    review_dir = RESULTS_DIR / "review" / run_id; review_dir.mkdir(parents=True, exist_ok=True)
    review = _render_review(select_review_cases(index), review_dir)
    verification = verify_automated(RESULTS_DIR, spec)
    city_summary = pd.read_csv(audit_dir / "summary_by_city.csv", encoding="utf-8-sig")
    write_researcher_summary(global_summary, city_summary, verification,
                             audit_dir / "researcher_summary.md", expected_entries=spec["expected_entries"])
    scripts = sorted(PHASE_DIR.glob("*.py"))
    output_paths = [RESULTS_DIR / "dataset/graph_dataset_index.csv.gz", audit_dir / "global_summary.json",
                    audit_dir / "verification_summary.json", audit_dir / "researcher_summary.md",
                    review_dir / "review_manifest.csv"]
    freeze = {
        "experiment_id": spec["experiment_id"], "run_id": run_id, "finished_at": datetime.now(timezone.utc).isoformat(),
        "command": "run_phase5.py --all-parts --run-id full_v01",
        "input_snapshot_sha256": sha256_file(RESULTS_DIR / "input_snapshot.json"),
        "script_sha256": {path.name: sha256_file(path) for path in scripts},
        "output_sha256": {str(path.relative_to(RESULTS_DIR)).replace("\\", "/"): sha256_file(path) for path in output_paths},
        "environment": json.loads((RESULTS_DIR / "environment.json").read_text(encoding="utf-8")),
        "evidence_state": verification["evidence_state"],
    }
    _atomic_json(RESULTS_DIR / "freeze.json", freeze)
    return {"passed": verification["automated_audit_passed"], "parts": part_results,
            "global_summary": global_summary, "review_cases": len(review), "verification": verification}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--part-id", choices=[f"part_{index:02d}" for index in range(1, 13)])
    mode.add_argument("--all-parts", action="store_true")
    parser.add_argument("--run-id", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        result = run_phase5(args.part_id, args.all_parts, args.run_id)
    except Exception as error:
        print(f"Phase 5 stopped: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
