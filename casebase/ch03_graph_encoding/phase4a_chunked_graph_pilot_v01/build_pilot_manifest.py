"""Build deterministic Phase 4A pilot and smoke manifests."""

from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


PHASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PHASE_DIR / "results"
SOURCE_MANIFEST = (
    PHASE_DIR.parent
    / "phase3_5b_final_sample_freeze_v01"
    / "results"
    / "dataset"
    / "graph_encoding_manifest.csv.gz"
)
SPEC_PATH = PHASE_DIR / "pilot_spec_v0.1.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assign_city_count_strata(manifest: pd.DataFrame) -> pd.DataFrame:
    """Assign four stable equal-frequency building-count strata within each city."""
    required = {"city", "block_id", "lod1_building_count"}
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(f"Manifest is missing columns: {', '.join(missing)}")
    if manifest[["city", "block_id"]].duplicated().any():
        raise ValueError("Manifest contains duplicate city + block_id keys")
    result = manifest.copy()
    result["lod1_building_count"] = pd.to_numeric(
        result["lod1_building_count"], errors="raise"
    ).astype(int)
    result = result.sort_values(
        ["city", "lod1_building_count", "block_id"], kind="mergesort"
    ).reset_index(drop=True)
    position = result.groupby("city", sort=False).cumcount()
    population = result.groupby("city", sort=False)["block_id"].transform("size")
    result["city_count_stratum"] = ((position * 4) // population).clip(upper=3) + 1
    return result


def build_pilot_manifest(
    manifest: pd.DataFrame,
    per_stratum: int,
    seed: int,
) -> pd.DataFrame:
    """Select a deterministic probability sample from every city × stratum cell."""
    if per_stratum <= 0:
        raise ValueError("per_stratum must be positive")
    assigned = assign_city_count_strata(manifest)
    selected: list[pd.DataFrame] = []
    for (city, stratum), group in assigned.groupby(
        ["city", "city_count_stratum"], sort=True
    ):
        if len(group) < per_stratum:
            raise ValueError(f"{city} stratum {stratum} has only {len(group)} rows")
        ranked = group.copy()
        ranked["_selection_key"] = ranked["block_id"].map(
            lambda block_id: hashlib.sha256(
                f"{seed}|{city}|{stratum}|{block_id}".encode("utf-8")
            ).hexdigest()
        )
        chosen = ranked.sort_values(
            ["_selection_key", "block_id"], kind="mergesort"
        ).head(per_stratum).copy()
        chosen["stratum_population"] = int(len(group))
        chosen["sample_weight"] = float(len(group) / per_stratum)
        selected.append(chosen.drop(columns="_selection_key"))
    pilot = pd.concat(selected, ignore_index=True)
    pilot = pilot.sort_values(
        ["city", "city_count_stratum", "lod1_building_count", "block_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    pilot.insert(0, "sample_id", pilot["block_id"].map(lambda value: f"P4A_{value}"))
    pilot["manifest_sha256"] = json.loads(SPEC_PATH.read_text(encoding="utf-8"))[
        "source_manifest_sha256"
    ]
    pilot["selection_seed"] = int(seed)
    return pilot


def build_smoke_manifest(pilot: pd.DataFrame) -> pd.DataFrame:
    """Choose the first building-count-ordered case in every pilot cell."""
    ordered = pilot.sort_values(
        ["city", "city_count_stratum", "lod1_building_count", "block_id"],
        kind="mergesort",
    )
    return (
        ordered.groupby(["city", "city_count_stratum"], sort=True, as_index=False)
        .head(1)
        .reset_index(drop=True)
    )


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig", lineterminator="\n", float_format="%.15g")


def main() -> int:
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    actual_hash = sha256_file(SOURCE_MANIFEST)
    if actual_hash != spec["source_manifest_sha256"]:
        raise RuntimeError(f"Frozen Phase 3.5B manifest SHA256 changed: {actual_hash}")
    manifest = pd.read_csv(
        SOURCE_MANIFEST,
        encoding="utf-8-sig",
        float_precision="round_trip",
        low_memory=False,
    )
    if len(manifest) != spec["source_manifest_rows"] or manifest["city"].nunique() != spec["city_count"]:
        raise RuntimeError("Frozen Phase 3.5B manifest size or city count changed")
    targets = [
        RESULTS_DIR / "pilot_manifest.csv",
        RESULTS_DIR / "smoke_manifest.csv",
        RESULTS_DIR / "input_snapshot.json",
        RESULTS_DIR / "environment.json",
    ]
    occupied = [path for path in targets if path.exists()]
    if occupied:
        raise FileExistsError(f"Refusing to overwrite Phase 4A manifests: {occupied}")
    pilot = build_pilot_manifest(
        manifest,
        per_stratum=int(spec["pilot_per_city_stratum"]),
        seed=int(spec["selection_seed"]),
    )
    smoke = build_smoke_manifest(pilot)
    if len(pilot) != spec["pilot_rows"] or len(smoke) != spec["smoke_rows"]:
        raise RuntimeError("Pilot or smoke sample did not close to its frozen size")
    _write_csv(targets[0], pilot)
    _write_csv(targets[1], smoke)
    captured = datetime.now(timezone.utc).isoformat()
    targets[2].write_text(
        json.dumps(
            {
                "captured_at_utc": captured,
                "source_manifest_path": str(SOURCE_MANIFEST.resolve()),
                "source_manifest_sha256": actual_hash,
                "source_manifest_rows": len(manifest),
                "source_city_count": int(manifest["city"].nunique()),
                "pilot_rows": len(pilot),
                "smoke_rows": len(smoke),
                "pilot_manifest_sha256": sha256_file(targets[0]),
                "smoke_manifest_sha256": sha256_file(targets[1]),
            },
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    targets[3].write_text(
        json.dumps(
            {
                "captured_at_utc": captured,
                "python": platform.python_version(),
                "pandas": pd.__version__,
                "platform": platform.platform(),
            },
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"pilot_rows": len(pilot), "smoke_rows": len(smoke)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
