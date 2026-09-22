"""Exclude only the two frozen classes of building-level polygon failures."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

from shapely import wkt
from shapely.geometry.base import BaseGeometry
from shapely.validation import make_valid


PHASE_DIR = Path(__file__).resolve().parent
PHASE25_DIR = PHASE_DIR.parent / "phase2_5_tessellation_relations_v02"
sys.path.insert(0, str(PHASE25_DIR))

from tessellation_encoding import _polygonal_geometry


def _source_geometry(record: dict[str, Any]) -> BaseGeometry:
    geometry = record.get("geometry")
    return geometry if geometry is not None else wkt.loads(record["geometry_wkt_metric"])


def _metric(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def _audit_exclusion(
    source: dict[str, Any],
    geometry: BaseGeometry,
    block_geometry: BaseGeometry,
    reason: str,
    validity_detail: str,
) -> dict[str, Any]:
    area = _metric(geometry.area) if geometry is not None else None
    perimeter = _metric(geometry.length) if geometry is not None else None
    compactness = (
        4.0 * math.pi * area / (perimeter * perimeter)
        if area is not None and perimeter is not None and area > 0 and perimeter > 0
        else None
    )
    covered = bool(block_geometry.covers(geometry)) if geometry is not None and not geometry.is_empty else False
    return {
        "building_id": str(source["building_id"]),
        "reason": reason,
        "footprint_area_m2": area,
        "footprint_perimeter_m": perimeter,
        "footprint_compactness": compactness,
        "geometry_was_valid": bool(geometry.is_valid) if geometry is not None else False,
        "geometry_validity_detail": validity_detail,
        "footprint_covered_by_block": covered,
    }


def partition_buildings_for_encoding(
    building_records: list[dict[str, Any]],
    block_geometry: BaseGeometry,
    shrink_m: float = 0.4,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if shrink_m < 0:
        raise ValueError("shrink_m must be non-negative")
    if block_geometry is None or block_geometry.is_empty:
        raise ValueError("block_geometry must be non-empty")
    block_polygon = _polygonal_geometry(
        make_valid(block_geometry) if not block_geometry.is_valid else block_geometry
    )
    if block_polygon.is_empty or block_polygon.area <= 0:
        raise ValueError("block_geometry must have positive polygonal area")

    kept: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for source in building_records:
        raw = _source_geometry(source)
        repaired = make_valid(raw) if not raw.is_valid else raw
        polygon = _polygonal_geometry(repaired)
        if polygon.is_empty or polygon.area <= 0:
            excluded.append(
                _audit_exclusion(
                    source,
                    raw,
                    block_polygon,
                    "invalid_polygonal_area_before_shrink",
                    "polygonal extraction after make_valid was empty or non-positive",
                )
            )
            continue
        shrunken = _polygonal_geometry(polygon.buffer(-shrink_m))
        if shrunken.is_empty or shrunken.area <= 0:
            excluded.append(
                _audit_exclusion(
                    source,
                    polygon,
                    block_polygon,
                    "empty_after_shrink",
                    "valid polygonal footprint became empty or non-positive after inward buffer",
                )
            )
            continue
        kept.append({**source, "geometry": polygon})
    return kept, excluded
