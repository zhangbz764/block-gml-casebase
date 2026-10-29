"""Freeze the authoritative Phase 4B full manifest and twelve manual parts."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
SOURCE = RESULTS_DIR / "metadata/screening/final/graph_encoding_manifest.csv.gz"
SPEC = json.loads((ROOT / "specs/encoding_spec_v032.json").read_text(encoding="utf-8"))
GROUP_SPEC = ROOT / "specs/city_group_spec_v01.csv"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_group_spec() -> pd.DataFrame:
    return pd.read_csv(GROUP_SPEC, encoding="utf-8-sig")


def assign_parts(manifest: pd.DataFrame, group_spec: pd.DataFrame) -> pd.DataFrame:
    result = manifest.copy()
    if result[["city", "block_id"]].duplicated().any():
        raise ValueError("Source manifest contains duplicate keys")
    city_to_part: dict[str, str] = {}
    for row in group_spec.itertuples(index=False):
        if row.subset_rule == "whole_city":
            for city in str(row.cities).split(";"):
                if city in city_to_part:
                    raise ValueError(f"City assigned twice: {city}")
                city_to_part[city] = row.part_id
    result["part_id"] = result["city"].map(city_to_part)
    tokyo_index = result.loc[result["city"].eq("Tokyo")].sort_values("block_id", kind="mergesort").index
    for position, index in enumerate(tokyo_index, start=1):
        result.at[index, "part_id"] = "part_01" if position % 2 else "part_02"
    if result["part_id"].isna().any():
        missing = sorted(result.loc[result["part_id"].isna(), "city"].unique())
        raise ValueError(f"Cities missing from group spec: {missing}")
    result.insert(0, "sample_id", result["block_id"].map(lambda value: f"P4B_{value}"))
    result["encoding_version"] = SPEC["encoding_version"]
    result["source_manifest_sha256"] = SPEC["source_manifest_sha256"]
    return result.sort_values(["part_id", "city", "block_id"], kind="mergesort").reset_index(drop=True)


def _stable_gzip(path: Path, frame: pd.DataFrame) -> None:
    payload = ("\ufeff" + frame.to_csv(index=False, lineterminator="\n", float_format="%.15g")).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            compressed.write(payload)


def build() -> int:
    if sha256_file(SOURCE) != SPEC["source_manifest_sha256"]:
        raise RuntimeError("Phase 3.5B source manifest SHA256 changed")
    source = pd.read_csv(SOURCE, encoding="utf-8-sig", low_memory=False, float_precision="round_trip")
    if len(source) != SPEC["source_manifest_rows"] or source["city"].nunique() != 24:
        raise RuntimeError("Phase 3.5B manifest row or city count changed")
    groups = load_group_spec()
    full = assign_parts(source, groups)
    actual = full.groupby("part_id").size().to_dict()
    expected = groups.set_index("part_id")["expected_block_count"].astype(int).to_dict()
    if actual != expected or sum(actual.values()) != SPEC["source_manifest_rows"]:
        raise RuntimeError(f"Part counts changed: {actual}")
    dataset = RESULTS_DIR / "dataset"
    full_path = dataset / "full_run_manifest_v032.csv.gz"
    part_paths = {part: dataset / "parts" / f"{part}_manifest.csv.gz" for part in expected}
    occupied = [path for path in [full_path, *part_paths.values()] if path.exists()]
    if occupied:
        raise FileExistsError(f"Refusing to overwrite frozen manifests: {occupied}")
    _stable_gzip(full_path, full)
    for part, path in part_paths.items():
        _stable_gzip(path, full.loc[full["part_id"].eq(part)].reset_index(drop=True))
    audit = RESULTS_DIR / "audit" / "prep"
    audit.mkdir(parents=True, exist_ok=True)
    summary = groups.copy()
    summary["actual_block_count"] = summary["part_id"].map(actual)
    summary.to_csv(audit / "city_group_summary.csv", index=False, encoding="utf-8-sig", lineterminator="\n")
    hashes = {
        "source_manifest_sha256": SPEC["source_manifest_sha256"],
        "full_run_manifest_v032.csv.gz": sha256_file(full_path),
        "parts": {part: sha256_file(path) for part, path in part_paths.items()},
    }
    (audit / "manifest_hashes.json").write_text(json.dumps(hashes, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(full), "parts": actual}, ensure_ascii=False))
    return 0



def main():
    global SOURCE,RESULTS_DIR
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=SOURCE)
    p.add_argument('--output',type=Path,required=True,help='new run directory')
    args=p.parse_args()
    if args.output.exists(): raise FileExistsError('output already exists: '+str(args.output))
    SOURCE=args.source.resolve();RESULTS_DIR=args.output.resolve()
    return build()

if __name__=='__main__': raise SystemExit(main())
