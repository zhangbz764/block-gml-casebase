"""Build one recoverable Phase 5 index shard from immutable Phase 4B outputs."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd

from graph_audit import FeatureAccumulator, audit_graph_record
from registry import sha256_file


PHASE_DIR = Path(__file__).resolve().parent
PHASE4B_DIR = PHASE_DIR.parent / "phase4b_full_graph_batches_v032"
PHASE5_AUDIT_CODE_FILES = (
    "build_graph_index.py", "graph_audit.py", "merge_graph_audit.py", "phase4b_gate.py",
    "registry.py", "render_visual_review.py", "run_phase5.py", "select_review_sample.py",
    "verify_phase5.py", "write_researcher_summary.py",
)


def phase5_code_bundle_sha256(paths: list[Path] | tuple[Path, ...] | None = None) -> str:
    selected = list(paths) if paths is not None else [PHASE_DIR / name for name in PHASE5_AUDIT_CODE_FILES]
    digest = hashlib.sha256()
    for path in sorted(selected, key=lambda item: item.name):
        digest.update(path.name.encode("utf-8")); digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii")); digest.update(b"\n")
    return digest.hexdigest()


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_graph(path: Path) -> dict:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    return json.loads(path.read_text(encoding="utf-8"))


def _latest_status(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    frame = frame.copy()
    frame["attempt"] = pd.to_numeric(frame["attempt"], errors="raise").astype(int)
    return frame.sort_values(["sample_id", "attempt"], kind="mergesort").drop_duplicates("sample_id", keep="last")


def _write_csv(path: Path, frame: pd.DataFrame, gzip_output: bool = False) -> Path:
    temporary = path.with_name(path.name + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    options = {"compression": {"method": "gzip", "mtime": 0}} if gzip_output else {}
    frame.to_csv(temporary, index=False, encoding="utf-8-sig", lineterminator="\n", **options)
    return temporary


def _write_json_temp(path: Path, value: dict) -> Path:
    temporary = path.with_name(path.name + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return temporary


def _issue(row: dict, issue_type: str, message: str) -> dict:
    return {
        "sample_id": row.get("sample_id", ""), "city": row.get("city", ""),
        "block_id": row.get("block_id", ""), "part_id": row.get("part_id", ""),
        "issue_type": issue_type, "message": message,
    }


def _excluded_index_row(manifest_row: dict, status: dict, metadata: dict) -> dict:
    return {
        "sample_id": status.get("sample_id", manifest_row.get("sample_id", "")),
        "city": status.get("city", manifest_row.get("city", "")),
        "country": manifest_row.get("country", ""), "block_id": status.get("block_id", manifest_row.get("block_id", "")),
        "part_id": status.get("part_id", manifest_row.get("part_id", "")), "case_status": status.get("status"),
        "graph_path": "", "encoding_version": metadata.get("encoding_version"),
        "source_manifest_sha256": metadata.get("source_manifest_sha256"),
        "full_manifest_sha256": metadata.get("full_manifest_sha256"),
        "part_manifest_sha256": metadata.get("part_manifest_sha256"),
        "code_bundle_sha256": metadata.get("code_bundle_sha256"), "parameter_sha256": metadata.get("parameter_sha256"),
        "graph_exists": False, "graph_readable": False, "file_bytes": 0,
        "validation_passed": False, "audit_passed": status.get("status") == "excluded_no_valid_building",
        "audit_issue_count": 0 if status.get("status") == "excluded_no_valid_building" else 1,
        "source_building_count": int(float(status.get("planned_building_count") or manifest_row.get("lod1_building_count") or 0)),
        "encoded_building_count": 0,
        "excluded_building_count": int(float(status.get("excluded_building_count") or status.get("planned_building_count") or 0)),
        "node_count": None, "cell_count": None, "cell_failure_count": None, "edge_count": None,
        "connected_components": None, "isolated_node_count": None, "average_degree": None, "edge_density": None,
        "long_distance_candidate_count": None, "low_shared_candidate_count": None, "combined_weak_candidate_count": None,
        "long_distance_candidate_ratio": None, "low_shared_candidate_ratio": None, "combined_weak_candidate_ratio": None,
        "excluded_building_ratio": 1.0, "isolated_node_ratio": None,
        **{field: manifest_row.get(field) for field in (
            "area_m2", "elongation", "compactness", "lod1_building_count", "bcr", "far", "avh_w", "hcv", "orientation_entropy"
        )},
    }


def build_part_index(part_id: str, input_root: Path, output_root: Path, spec: dict) -> dict:
    started = _utc()
    manifest_path = input_root / "results" / "dataset" / "parts" / f"{part_id}_manifest.csv.gz"
    run_dir = input_root / "results" / "runs" / "full_v032" / part_id
    status_path = run_dir / "encoding_status.csv"
    preflight_path = run_dir / "preflight.json"
    for required in (manifest_path, status_path, preflight_path):
        if not required.is_file():
            raise FileNotFoundError(required)
    metadata = json.loads(preflight_path.read_text(encoding="utf-8")).get("phase4b", {})
    if metadata.get("part_id") != part_id:
        raise RuntimeError(f"Preflight part mismatch for {part_id}")
    if metadata.get("part_manifest_sha256") != sha256_file(manifest_path):
        raise RuntimeError(f"Frozen manifest SHA256 mismatch for {part_id}")
    if metadata.get("code_bundle_sha256") != spec["phase4b_code_bundle_sha256"]:
        raise RuntimeError("Phase 4B code bundle differs from the Phase 5 audit specification")
    if metadata.get("source_manifest_sha256") != spec["phase35b_manifest_sha256"]:
        raise RuntimeError("Phase 3.5B source manifest differs from the Phase 5 audit specification")
    if metadata.get("full_manifest_sha256") != spec["phase4b_full_manifest_sha256"]:
        raise RuntimeError("Phase 4B full manifest differs from the Phase 5 audit specification")

    audit_spec_path = Path(spec.get("_spec_path", PHASE_DIR / "audit_spec_v01.json"))
    phase5_bundle = str(spec.get("phase5_code_bundle_sha256") or phase5_code_bundle_sha256())
    signature = {
        "part_id": part_id, "input_manifest_sha256": sha256_file(manifest_path),
        "status_ledger_sha256": sha256_file(status_path), "preflight_sha256": sha256_file(preflight_path),
        "audit_spec_sha256": sha256_file(audit_spec_path), "phase4b_code_bundle_sha256": metadata["code_bundle_sha256"],
        "phase5_code_bundle_sha256": phase5_bundle,
    }
    dataset_dir = output_root / "dataset" / "index_parts"
    audit_dir = output_root / "audit" / spec.get("formal_output_name", "full_v01") / "parts"
    index_path = dataset_dir / f"{part_id}_graph_index.csv.gz"
    feature_path = audit_dir / f"{part_id}_feature_counts.csv"
    issue_path = audit_dir / f"{part_id}_issues.csv"
    meta_path = audit_dir / f"{part_id}_audit_meta.json"
    if meta_path.is_file():
        existing = json.loads(meta_path.read_text(encoding="utf-8"))
        if any(existing.get(key) != value for key, value in signature.items()):
            raise RuntimeError(f"Existing {part_id} shard has incompatible inputs")
        paths = {"index": index_path, "features": feature_path, "issues": issue_path}
        if not all(path.is_file() and sha256_file(path) == existing.get("output_sha256", {}).get(name) for name, path in paths.items()):
            raise RuntimeError(f"Existing {part_id} shard is incomplete or changed")
        return {**existing, "skipped": True}

    manifest = pd.read_csv(manifest_path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    expected_part_rows = spec.get("part_entry_counts", {}).get(part_id)
    if expected_part_rows is not None and len(manifest) != int(expected_part_rows):
        raise RuntimeError(f"{part_id} manifest has {len(manifest)} rows, expected {expected_part_rows}")
    statuses = _latest_status(pd.read_csv(status_path, encoding="utf-8-sig", dtype=str, keep_default_na=False))
    latest = {str(row["sample_id"]): row for row in statuses.to_dict("records")}
    rows: list[dict] = []
    issues: list[dict] = []
    accumulator = FeatureAccumulator()
    counts = {"completed": 0, "excluded_no_valid_building": 0, "failed": 0, "interrupted": 0, "missing": 0}
    manifest_ids = set(manifest["sample_id"].astype(str))
    for manifest_row in manifest.to_dict("records"):
        status = latest.get(str(manifest_row["sample_id"]))
        if status is None:
            status = {**manifest_row, "part_id": part_id, "status": "missing"}
        case_status = str(status.get("status", "missing"))
        counts[case_status] = counts.get(case_status, 0) + 1
        if case_status == "completed":
            raw = Path(str(status.get("graph_path", "")))
            graph_path = (input_root / raw).resolve()
            try:
                graph_path.relative_to(input_root.resolve())
                if not graph_path.is_file():
                    raise FileNotFoundError(graph_path)
                record = _read_graph(graph_path)
                row, graph_issues = audit_graph_record(record, status, metadata)
                row["file_bytes"] = graph_path.stat().st_size
                rows.append(row); issues.extend(graph_issues)
                accumulator.update(record, row)
                if accumulator.issues:
                    row["audit_issue_count"] += len(accumulator.issues)
                    row["audit_passed"] = False
                    issues.extend(accumulator.issues)
                    accumulator.issues.clear()
            except Exception as error:
                row = _excluded_index_row(manifest_row, status, metadata)
                row.update({"graph_path": status.get("graph_path", ""), "graph_exists": graph_path.is_file(),
                            "case_status": case_status, "audit_passed": False, "audit_issue_count": 1})
                rows.append(row)
                issues.append(_issue(row, "graph_read_failed", f"{type(error).__name__}: {error}"))
        else:
            row = _excluded_index_row(manifest_row, status, metadata)
            rows.append(row)
            if case_status == "excluded_no_valid_building" and str(status.get("graph_path", "")).strip():
                row["audit_passed"] = False; row["audit_issue_count"] = 1
                issues.append(_issue(row, "excluded_case_has_graph_path", "Excluded entry must not claim a Graph path"))
            elif case_status != "excluded_no_valid_building":
                issues.append(_issue(row, f"unclosed_status_{case_status}", f"Entry has terminal status {case_status!r}"))
    for extra in sorted(set(latest) - manifest_ids):
        issues.append(_issue(latest[extra], "status_not_in_manifest", "Status row is absent from the frozen part manifest"))

    index = pd.DataFrame(rows)
    if len(index) != len(manifest):
        raise RuntimeError("Part index row count changed during construction")
    feature_rows = accumulator.to_rows()
    for city in sorted(index["city"].astype(str).unique()):
        feature_rows.extend(accumulator.to_rows(city))
    feature_frame = pd.DataFrame(feature_rows)
    issue_frame = pd.DataFrame(issues, columns=["sample_id", "city", "block_id", "part_id", "issue_type", "message"])

    temporary = {
        "index": _write_csv(index_path, index, gzip_output=True),
        "features": _write_csv(feature_path, feature_frame),
        "issues": _write_csv(issue_path, issue_frame),
    }
    output_hashes = {name: sha256_file(path) for name, path in temporary.items()}
    meta = {
        **signature, "started_at": started, "finished_at": _utc(), "manifest_rows": len(manifest),
        "index_rows": len(index), "latest_status_rows": len(statuses), "status_counts": counts,
        "graphs_read": counts["completed"], "issue_count": len(issue_frame),
        "accumulated_objects": int(feature_frame.loc[feature_frame["scope"].eq("__global__"), "object_count"].sum()),
        "output_sha256": output_hashes, "passed": len(issue_frame) == 0,
    }
    meta_temp = _write_json_temp(meta_path, meta)
    try:
        for name, destination in (("features", feature_path), ("issues", issue_path), ("index", index_path)):
            os.replace(temporary[name], destination)
        os.replace(meta_temp, meta_path)
    finally:
        for path in [*temporary.values(), meta_temp]:
            if path.exists():
                path.unlink()
    return {**meta, "skipped": False}
