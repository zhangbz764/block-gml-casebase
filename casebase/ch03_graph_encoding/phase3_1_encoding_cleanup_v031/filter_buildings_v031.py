"""Pre-filter building footprints that collapse under the fixed v0.3 buffer."""

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


def _building_geometry(record: dict[str, Any]) -> BaseGeometry:
    geometry = record.get("geometry")
    if geometry is None:
        geometry = wkt.loads(record["geometry_wkt_metric"])
    if not geometry.is_valid:
        geometry = _polygonal_geometry(make_valid(geometry))
    else:
        geometry = _polygonal_geometry(geometry)
    if geometry.is_empty or geometry.area <= 0:
        raise ValueError(
            f"Building {record.get('building_id')} has no valid polygonal area before shrink"
        )
    return geometry


def partition_buildings_for_encoding(
    building_records: list[dict[str, Any]],
    block_geometry: BaseGeometry,
    shrink_m: float = 0.4,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return encodable buildings and collapsed-footprint audit records."""
    if shrink_m < 0:
        raise ValueError("shrink_m must be non-negative")
    if block_geometry is None or block_geometry.is_empty:
        raise ValueError("block_geometry must be non-empty")
    if not block_geometry.is_valid:
        block_geometry = _polygonal_geometry(make_valid(block_geometry))
    else:
        block_geometry = _polygonal_geometry(block_geometry)
    if block_geometry.is_empty or block_geometry.area <= 0:
        raise ValueError("block_geometry must have positive polygonal area")

    kept: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for source in building_records:
        geometry = _building_geometry(source)
        shrunken = _polygonal_geometry(geometry.buffer(-shrink_m))
        if shrunken.is_empty or shrunken.area <= 0:
            perimeter = float(geometry.length)
            compactness = (
                4.0 * math.pi * float(geometry.area) / (perimeter * perimeter)
                if perimeter > 0
                else None
            )
            excluded.append(
                {
                    "building_id": str(source["building_id"]),
                    "reason": "empty_after_shrink",
                    "footprint_area_m2": float(geometry.area),
                    "footprint_perimeter_m": perimeter,
                    "footprint_compactness": compactness,
                    "footprint_covered_by_block": bool(block_geometry.covers(geometry)),
                }
            )
            continue
        kept.append({**source, "geometry": geometry})
    return kept, excluded
