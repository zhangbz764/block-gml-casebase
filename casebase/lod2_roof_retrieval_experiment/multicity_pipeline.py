"""Pure selection and hard-filter rules for the multicity LOD2 experiment."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from block_metrics import BUILDING_METRIC_COLUMNS, BLOCK_METRIC_COLUMNS, ROOF_FEATURES


FROZEN_M1_FEATURES = [
    "log_block_area",
    "compactness",
    "elongation",
    "bcr",
    "mean_h_g1",
    "h_g1_cv",
    "log_building_count",
    "mean_footprint_area",
    "footprint_area_cv",
    "orientation_entropy",
    "normalized_nn_distance",
]

SAMPLE_MANIFEST_COLUMNS = [
    "block_id",
    "random_order",
    "processed",
    "data_valid",
    "in_reference_scope",
    "is_eligible",
    "selected",
    "exclusion_codes",
]
PROCESSED_BLOCK_COLUMNS = [
    *BLOCK_METRIC_COLUMNS,
    "data_valid",
    "in_reference_scope",
    "is_eligible",
    "exclusion_codes",
    "random_order",
    "selected",
]

CHECKPOINT_COLUMNS = {
    "manifest": SAMPLE_MANIFEST_COLUMNS,
    "processed_blocks": PROCESSED_BLOCK_COLUMNS,
    "selected_blocks": PROCESSED_BLOCK_COLUMNS,
    "selected_buildings": BUILDING_METRIC_COLUMNS,
}


@dataclass(frozen=True)
class CitySampleResult:
    manifest: pd.DataFrame
    processed_blocks: pd.DataFrame
    selected_blocks: pd.DataFrame
    selected_buildings: pd.DataFrame


def load_city_checkpoint(
    paths: Mapping[str, Path], expected_fingerprint: str
) -> dict[str, pd.DataFrame] | None:
    """Load a complete compatible checkpoint, otherwise request recomputation."""

    required_keys = {*CHECKPOINT_COLUMNS, "metadata"}
    if required_keys - set(paths) or any(
        not Path(paths[key]).is_file() for key in required_keys
    ):
        return None
    try:
        metadata = json.loads(Path(paths["metadata"]).read_text(encoding="utf-8"))
        if metadata.get("fingerprint") != expected_fingerprint:
            return None
        frames = {
            key: pd.read_parquet(Path(paths[key])) for key in CHECKPOINT_COLUMNS
        }
        for key, columns in CHECKPOINT_COLUMNS.items():
            if set(columns) - set(frames[key].columns):
                return None
        return frames
    except Exception:
        return None


def write_city_checkpoint(
    paths: Mapping[str, Path],
    frames: Mapping[str, pd.DataFrame],
    metadata: Mapping,
) -> None:
    """Write checkpoint data first and publish its compatibility marker last."""

    required_keys = {*CHECKPOINT_COLUMNS, "metadata"}
    missing_paths = required_keys - set(paths)
    missing_frames = set(CHECKPOINT_COLUMNS) - set(frames)
    if missing_paths or missing_frames:
        raise ValueError(
            f"Incomplete checkpoint arguments: paths={sorted(missing_paths)}, "
            f"frames={sorted(missing_frames)}"
        )
    for key, columns in CHECKPOINT_COLUMNS.items():
        missing = set(columns) - set(frames[key].columns)
        if missing:
            raise ValueError(f"Checkpoint frame {key} missing columns: {sorted(missing)}")

    resolved = {key: Path(value) for key, value in paths.items()}
    for path in resolved.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    temporary = {
        key: path.with_name(f"{path.name}.tmp") for key, path in resolved.items()
    }
    try:
        for key in CHECKPOINT_COLUMNS:
            frames[key].to_parquet(temporary[key], index=False)
        temporary["metadata"].write_text(
            json.dumps(dict(metadata), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        resolved["metadata"].unlink(missing_ok=True)
        for key in CHECKPOINT_COLUMNS:
            os.replace(temporary[key], resolved[key])
        os.replace(temporary["metadata"], resolved["metadata"])
    finally:
        for path in temporary.values():
            path.unlink(missing_ok=True)


def config_fingerprint(config: dict) -> str:
    """Return a short stable digest for checkpoint compatibility checks."""

    payload = json.dumps(
        config, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def id_set_fingerprint(values: Iterable[str]) -> str:
    """Digest a set of source IDs so changed candidate provenance invalidates caches."""

    digest = hashlib.sha256()
    for value in sorted({str(item) for item in values}):
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, byteorder="big"))
        digest.update(encoded)
    return digest.hexdigest()[:16]


def stable_city_seed(base_seed: int, city: str) -> int:
    """Derive a stable NumPy seed without relying on Python's randomized hash."""

    normalized = str(city).strip().lower()
    if not normalized:
        raise ValueError("city cannot be empty")
    digest = hashlib.sha256(f"{int(base_seed)}:{normalized}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="big", signed=False)


def randomized_block_ids(
    block_ids: Iterable[str], base_seed: int, city: str
) -> list[str]:
    """Return a reproducible permutation independent of database row order."""

    unique = sorted({str(value) for value in block_ids})
    if not unique:
        return []
    generator = np.random.default_rng(stable_city_seed(base_seed, city))
    positions = generator.permutation(len(unique))
    return [unique[int(position)] for position in positions]


def _append_code(codes: list[str], code: str) -> None:
    if code and code not in codes:
        codes.append(code)


def classify_block_quality(
    frame: pd.DataFrame, min_bcr: float = 0.02
) -> pd.DataFrame:
    """Separate objective data failures from the low-BCR reference scope rule."""

    if not 0 <= min_bcr <= 1:
        raise ValueError("min_bcr must lie between zero and one")
    required = {
        "block_id",
        "area_m2",
        "compactness",
        "elongation",
        "bcr",
        "is_usable",
        *FROZEN_M1_FEATURES,
        *ROOF_FEATURES,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing hard-filter columns: {missing}")

    result = frame.copy()
    data_valid_values: list[bool] = []
    scope_values: list[bool] = []
    code_values: list[str] = []
    required_metrics = FROZEN_M1_FEATURES + list(ROOF_FEATURES)

    for row in result.to_dict(orient="records"):
        codes: list[str] = []
        if not bool(row["is_usable"]):
            _append_code(codes, "upstream_quality_failure")

        area = float(row["area_m2"])
        compactness = float(row["compactness"])
        elongation = float(row["elongation"])
        bcr = float(row["bcr"])
        slope = float(row["roof_mean_slope"])
        relief = float(row["roof_relief_ratio_median"])
        slope_sd = float(row["building_roof_slope_sd"])

        if not np.isfinite(area) or area <= 0:
            _append_code(codes, "invalid_block_area")
        if not np.isfinite(compactness) or not 0 < compactness <= 1:
            _append_code(codes, "invalid_compactness")
        if not np.isfinite(elongation) or not 0 < elongation <= 1:
            _append_code(codes, "invalid_elongation")
        if not np.isfinite(bcr) or bcr <= 0:
            _append_code(codes, "non_positive_bcr")
        elif bcr > 1:
            _append_code(codes, "bcr_above_one")
        if not np.isfinite(slope) or not 0 <= slope <= 90:
            _append_code(codes, "invalid_roof_slope")
        if not np.isfinite(relief) or not 0 <= relief <= 1:
            _append_code(codes, "invalid_roof_relief")
        if not np.isfinite(slope_sd) or slope_sd < 0:
            _append_code(codes, "invalid_roof_slope_dispersion")
        if any(not np.isfinite(float(row[name])) for name in required_metrics):
            _append_code(codes, "non_finite_required_metric")

        data_valid = not codes
        in_scope = data_valid and bcr >= min_bcr
        if data_valid and not in_scope:
            _append_code(codes, "below_reference_bcr")
        data_valid_values.append(data_valid)
        scope_values.append(in_scope)
        code_values.append(";".join(codes))

    result["data_valid"] = data_valid_values
    result["in_reference_scope"] = scope_values
    result["is_eligible"] = result["data_valid"] & result["in_reference_scope"]
    result["exclusion_codes"] = code_values
    return result


def take_first_eligible_in_order(
    frame: pd.DataFrame, ordered_ids: Sequence[str], target_count: int
) -> pd.DataFrame:
    """Take the first eligible rows in a precomputed random block order."""

    if target_count < 0:
        raise ValueError("target_count cannot be negative")
    required = {"block_id", "is_eligible"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing selection columns: {missing}")
    order = {str(block_id): index for index, block_id in enumerate(ordered_ids)}
    selected = frame.loc[frame["is_eligible"].astype(bool)].copy()
    selected["_random_order"] = selected["block_id"].astype(str).map(order)
    selected = selected.loc[selected["_random_order"].notna()]
    selected = selected.sort_values("_random_order", kind="mergesort").head(target_count)
    return selected.drop(columns="_random_order").reset_index(drop=True)


def collect_city_sample(
    ordered_ids: Sequence[str],
    measure_batch: Callable[[list[str]], tuple[pd.DataFrame, pd.DataFrame]],
    target_count: int = 5000,
    batch_size: int = 1000,
    min_bcr: float = 0.02,
) -> CitySampleResult:
    """Measure randomized batches until the requested eligible sample is full."""

    if target_count < 0:
        raise ValueError("target_count cannot be negative")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    ordered = [str(value) for value in ordered_ids]
    if len(ordered) != len(set(ordered)):
        raise ValueError("ordered_ids must be unique")
    order_lookup = {block_id: position for position, block_id in enumerate(ordered)}
    selected_ids: list[str] = []
    processed_parts: list[pd.DataFrame] = []
    selected_building_parts: list[pd.DataFrame] = []

    for start in range(0, len(ordered), batch_size):
        if len(selected_ids) >= target_count:
            break
        batch_ids = ordered[start : start + batch_size]
        block_frame, building_frame = measure_batch(batch_ids)
        measured_ids = set(block_frame["block_id"].astype(str))
        missing_ids = sorted(set(batch_ids) - measured_ids)
        if missing_ids:
            raise RuntimeError("Missing measured block rows: " + ", ".join(missing_ids))
        classified = classify_block_quality(block_frame, min_bcr=min_bcr)
        classified["random_order"] = (
            classified["block_id"].astype(str).map(order_lookup).astype(int)
        )
        remaining = target_count - len(selected_ids)
        newly_selected = take_first_eligible_in_order(
            classified, ordered_ids=ordered, target_count=remaining
        )["block_id"].astype(str).tolist()
        selected_ids.extend(newly_selected)
        classified["selected"] = classified["block_id"].astype(str).isin(newly_selected)
        processed_parts.append(classified)
        if newly_selected:
            if "block_id" not in building_frame.columns:
                raise ValueError("Measured building rows must contain block_id")
            selected_building_parts.append(
                building_frame.loc[
                    building_frame["block_id"].astype(str).isin(newly_selected)
                ].copy()
            )

    processed_blocks = (
        pd.concat(processed_parts, ignore_index=True)
        if processed_parts
        else pd.DataFrame(columns=PROCESSED_BLOCK_COLUMNS)
    )
    if not processed_blocks.empty:
        processed_blocks = processed_blocks.sort_values("random_order", kind="mergesort")
    selected_blocks = processed_blocks.loc[
        processed_blocks.get("selected", False).astype(bool)
    ].reset_index(drop=True)
    selected_buildings = (
        pd.concat(selected_building_parts, ignore_index=True)
        if selected_building_parts
        else pd.DataFrame(columns=BUILDING_METRIC_COLUMNS)
    )

    processed_lookup = {
        str(row.block_id): row
        for row in processed_blocks.itertuples(index=False)
    }
    manifest_rows = []
    for block_id in ordered:
        row = processed_lookup.get(block_id)
        manifest_rows.append(
            {
                "block_id": block_id,
                "random_order": order_lookup[block_id],
                "processed": row is not None,
                "data_valid": getattr(row, "data_valid", pd.NA),
                "in_reference_scope": getattr(row, "in_reference_scope", pd.NA),
                "is_eligible": getattr(row, "is_eligible", pd.NA),
                "selected": bool(getattr(row, "selected", False)),
                "exclusion_codes": getattr(row, "exclusion_codes", ""),
            }
        )
    manifest = pd.DataFrame(manifest_rows, columns=SAMPLE_MANIFEST_COLUMNS)
    return CitySampleResult(
        manifest=manifest,
        processed_blocks=processed_blocks.reset_index(drop=True),
        selected_blocks=selected_blocks,
        selected_buildings=selected_buildings,
    )
