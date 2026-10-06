"""Compact completed historical outputs without changing frozen encoding code.

Run with --apply to pack the explicit historical paths below. Restore a phase
with --restore PHASE before rerunning its legacy reporting/verification scripts.
No database access or graph construction is performed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import zipfile


CH03_DIR = Path(__file__).resolve().parents[1]
MANIFEST = "__manifest__.json"
RECORDS = {
    "phase2_encoding_v01": (
        "phase1_freeze_sha256.csv", "run_metadata.json", "validation_summary.csv",
    ),
    "phase2_5_tessellation_relations_v02": (
        "environment.json", "output_inventory.csv", "parameter_sensitivity.csv",
        "validation_issues.csv", "validation_summary.csv",
    ),
    "phase3_encoding_dataset_v03": (
        "environment.json", "output_inventory.csv", "verification_summary.json",
        "sampling_frame_summary.csv", "sampling_strata.csv", "sampling_supplementation.csv",
        "audit/audit_summary.json", "audit/feature_profile_by_city.csv",
        "audit/feature_profile_global.csv", "audit/feature_redundancy_pairs.csv",
        "audit/graph_structure_summary.csv", "audit/quality_issue_summary.csv",
        "audit/runtime_storage_summary.csv", "pilot/encoding_status.csv",
        "pilot/freeze.json", "pilot/pilot_manifest.csv", "pilot/pilot_summary.json",
        "visual_checks/city_anchor_selection.csv", "visual_checks/flagged_case_selection.csv",
    ),
    "phase3_1_encoding_cleanup_v031": ("environment.json", "output_inventory.csv"),
    "phase3_5a_block_screening_v01": (
        "environment.json", "pilot", "audit/pilot_v01",
        "audit/full_v01/filter_overlap_summary.csv",
        "audit/full_v01/filter_reason_summary.csv", "audit/full_v01/output_inventory.csv",
    ),
    "phase4a_chunked_graph_pilot_v01": ("audit/pilot_v01", "audit/smoke_v01"),
    "phase4b_full_graph_batches_v032": (
        "audit/prep/city_group_summary.csv", "audit/prep/verification_summary.json",
        *tuple(
            f"audit/full_v032/part_{part:02d}/{name}"
            for part in range(1, 13)
            for name in ("part_summary.json", "status_by_city.csv", "failure_summary.csv")
        ),
    ),
}
HISTORY = {
    "phase2_encoding_v01": ("visual_checks",),
    "phase2_5_tessellation_relations_v02": ("cells", "visual_checks", "visual_review.html"),
    "phase3_encoding_dataset_v03": ("pilot/run_log.txt", "visual_checks"),
    "phase3_5a_block_screening_v01": ("visual_review",),
    "phase4a_chunked_graph_pilot_v01": ("pilot", "smoke"),
}


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def checked_path(root: Path, name: str) -> Path:
    relative = PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts or "\\" in name or ":" in name:
        raise ValueError(f"Unsafe archive path: {name}")
    path = root.joinpath(*relative.parts).resolve()
    path.relative_to(root.resolve())
    return path


def selected_files(root: Path, names: tuple[str, ...]) -> list[Path]:
    files: set[Path] = set()
    for name in names:
        path = checked_path(root, name)
        if path.is_file():
            files.add(path)
        elif path.is_dir():
            for child in path.rglob("*"):
                if child.is_file():
                    files.add(checked_path(root, child.relative_to(root).as_posix()))
    return sorted(files)


def prune_parents(root: Path, paths: list[Path]) -> None:
    parents = {parent for path in paths for parent in path.parents if parent != root and root in parent.parents}
    for parent in sorted(parents, key=lambda item: len(item.parts), reverse=True):
        if parent.is_dir() and not any(parent.iterdir()):
            parent.rmdir()


def pack(root: Path, destination: Path, paths: list[Path]) -> dict:
    """Replace loose files only after their bytes have been verified in the ZIP."""
    root = root.resolve()
    destination = checked_path(root, destination.relative_to(root).as_posix())
    if not paths:
        return {"bundle": str(destination), "packed": 0}
    sources = {path.relative_to(root).as_posix(): path for path in paths}
    sources = {name: checked_path(root, name) for name in sources}
    code_extensions = {".py", ".ipynb", ".sql", ".ps1", ".sh", ".bat", ".cmd", ".r", ".js", ".ts"}
    if any(path.suffix.lower() in code_extensions for path in sources.values()):
        raise ValueError("Source code must remain loose and cannot be compacted")
    source_hashes = {name: digest(path) for name, path in sources.items()}
    previous: dict[str, dict] = {}
    if destination.is_file():
        with zipfile.ZipFile(destination) as archive:
            previous = json.loads(archive.read(MANIFEST))
            for name in previous:
                checked_path(root, name)
            if set(previous) != set(archive.namelist()) - {MANIFEST}:
                raise ValueError(f"Unexpected archive members: {destination}")
            for name in sources.keys() & previous.keys():
                if previous[name]["sha256"] != source_hashes[name]:
                    raise ValueError(f"New output differs from its archived version: {name}")
    entries = dict(previous)
    entries.update({name: {"bytes": path.stat().st_size, "sha256": source_hashes[name]} for name, path in sources.items()})
    temporary = destination.with_name(destination.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"Existing temporary archive must be inspected: {temporary}")
    try:
        with zipfile.ZipFile(temporary, "w", allowZip64=True) as output:
            old = zipfile.ZipFile(destination) if destination.is_file() else None
            try:
                for name in sorted(entries):
                    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                    info.compress_type = zipfile.ZIP_STORED if name.endswith((".gz", ".png", ".jpg")) else zipfile.ZIP_DEFLATED
                    source = sources[name].open("rb") if name in sources else old.open(name)
                    with source, output.open(info, "w", force_zip64=True) as handle:
                        shutil.copyfileobj(source, handle, length=1024 * 1024)
                output.writestr(MANIFEST, json.dumps(entries, ensure_ascii=False, indent=2) + "\n")
            finally:
                if old is not None:
                    old.close()
        with zipfile.ZipFile(temporary) as archive:
            for name, entry in entries.items():
                with archive.open(name) as handle:
                    if hashlib.file_digest(handle, "sha256").hexdigest() != entry["sha256"]:
                        raise ValueError(f"Archive verification failed: {name}")
        if any(digest(path) != source_hashes[name] for name, path in sources.items()):
            raise ValueError("A source file changed during archiving; no originals removed")
        os.replace(temporary, destination)
        for path in sources.values():
            path.unlink()
        prune_parents(root, list(sources.values()))
    finally:
        if temporary.exists():
            temporary.unlink()
    return {"bundle": str(destination), "packed": len(sources), "entries": len(entries), "bytes": destination.stat().st_size}


def restore(root: Path, archive_path: Path, prefix: str = "") -> int:
    """Restore exact bytes; refuse to overwrite a newer or manually edited file."""
    if not archive_path.is_file():
        return 0
    with zipfile.ZipFile(archive_path) as archive:
        entries = json.loads(archive.read(MANIFEST))
        if set(entries) != set(archive.namelist()) - {MANIFEST}:
            raise ValueError(f"Unexpected archive members: {archive_path}")
        paths = {name: checked_path(root, name) for name in entries if name.startswith(prefix)}
        for name, path in paths.items():
            if path.exists() and (not path.is_file() or digest(path) != entries[name]["sha256"]):
                raise ValueError(f"Refusing to overwrite changed file: {path}")
            with archive.open(name) as handle:
                if hashlib.file_digest(handle, "sha256").hexdigest() != entries[name]["sha256"]:
                    raise ValueError(f"Archive verification failed: {name}")
        count = 0
        for name, path in paths.items():
            if path.exists():
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(name) as source, path.open("xb") as target:
                shutil.copyfileobj(source, target, length=1024 * 1024)
            count += 1
        return count


def merged_part_reports(root: Path) -> None:
    """Expose two readable total tables while the exact old reports stay archived."""
    phase = root / "phase4b_full_graph_batches_v032" / "results"
    bundle = phase / "records.zip"
    if not bundle.is_file():
        return
    with zipfile.ZipFile(bundle) as archive:
        summaries: list[dict] = []
        city_rows: list[dict] = []
        city_fields: list[str] = ["part_id"]
        for part in range(1, 13):
            part_id = f"part_{part:02d}"
            base = f"audit/full_v032/{part_id}/"
            if base + "part_summary.json" not in archive.namelist():
                continue
            summary = json.loads(archive.read(base + "part_summary.json"))
            summaries.append({"part_id": part_id, **summary})
            reader = csv.DictReader(io.StringIO(archive.read(base + "status_by_city.csv").decode("utf-8-sig")))
            for field in reader.fieldnames or []:
                if field not in city_fields:
                    city_fields.append(field)
            city_rows.extend({"part_id": part_id, **row} for row in reader)
        audit = phase / "audit" / "full_v032"
        audit.mkdir(parents=True, exist_ok=True)
        for name, rows, fields in (
            ("part_summary.csv", summaries, list(dict.fromkeys(key for row in summaries for key in row))),
            ("status_by_city.csv", city_rows, city_fields),
        ):
            if not fields:
                continue
            target = audit / name
            temporary = target.with_name(target.name + ".tmp")
            try:
                with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=fields)
                    writer.writeheader()
                    writer.writerows(rows)
                os.replace(temporary, target)
            finally:
                if temporary.exists():
                    temporary.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--apply", action="store_true")
    modes.add_argument("--restore", metavar="PHASE", help="Phase directory name, or all")
    parser.add_argument("--bundle", choices=("records", "history", "handoff", "all"), default="all")
    args = parser.parse_args()
    if args.restore:
        available = {path.name for path in CH03_DIR.glob("phase*") if path.is_dir()}
        names = sorted(available) if args.restore == "all" else [args.restore]
        if any(name not in available for name in names):
            parser.error("Unknown historical phase")
        bundles = ("records", "history") if args.bundle == "all" else (() if args.bundle == "handoff" else (args.bundle,))
        for name in names:
            root = CH03_DIR / name / "results"
            for bundle in bundles:
                count = restore(root, root / f"{bundle}.zip")
                print(json.dumps({"phase": name, "bundle": bundle, "restored": count}, ensure_ascii=False))
            if args.bundle in ("handoff", "all"):
                count = restore(CH03_DIR, CH03_DIR / "evidence_pack.zip", prefix=f"{name}/")
                print(json.dumps({"phase": name, "bundle": "handoff", "restored": count}, ensure_ascii=False))
        return 0
    for mapping, bundle in ((RECORDS, "records"), (HISTORY, "history")):
        for name, selectors in mapping.items():
            root = (CH03_DIR / name / "results").resolve()
            files = selected_files(root, selectors)
            result = pack(root, root / f"{bundle}.zip", files) if args.apply else {"phase": name, "bundle": bundle, "files": len(files)}
            print(json.dumps(result, ensure_ascii=False))
    document_patterns = (
        "*_evidence_pack.md", "plan_review.md", "sampling_report.md",
        "*_decision_summary.md", "*_spot_checks.md",
    )
    documents = sorted({
        path for pattern in document_patterns
        for path in CH03_DIR.glob(f"phase*/results/**/{pattern}")
    })
    if args.apply:
        print(json.dumps(pack(CH03_DIR, CH03_DIR / "evidence_pack.zip", documents), ensure_ascii=False))
        merged_part_reports(CH03_DIR)
    else:
        print(json.dumps({"bundle": "evidence_pack.zip", "files": len(documents)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
