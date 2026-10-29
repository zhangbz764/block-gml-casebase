"""Morphological tessellation and relation encoding (original v0.2 algorithm)."""

from __future__ import annotations

import json
import math
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree
from shapely import make_valid, voronoi_polygons, wkt
from shapely.geometry import MultiPoint, MultiPolygon, Point, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import polygonize, unary_union


@dataclass(frozen=True)
class TessellationConfig:
    shrink_m: float = 0.4
    segment_m: float = 0.5
    min_shared_boundary_m: float = 1e-6

    def __post_init__(self) -> None:
        if self.shrink_m < 0 or not math.isfinite(self.shrink_m):
            raise ValueError("shrink_m must be finite and non-negative")
        if self.segment_m <= 0 or not math.isfinite(self.segment_m):
            raise ValueError("segment_m must be finite and greater than zero")
        if self.min_shared_boundary_m < 0 or not math.isfinite(
            self.min_shared_boundary_m
        ):
            raise ValueError("min_shared_boundary_m must be finite and non-negative")


def write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _polygon_parts(geometry: BaseGeometry) -> list[Polygon]:
    if isinstance(geometry, Polygon):
        return [geometry]
    if isinstance(geometry, MultiPolygon):
        return list(geometry.geoms)
    parts: list[Polygon] = []
    for item in getattr(geometry, "geoms", []):
        parts.extend(_polygon_parts(item))
    return parts


def _polygonal_geometry(geometry: BaseGeometry) -> BaseGeometry:
    parts = [part for part in _polygon_parts(geometry) if not part.is_empty and part.area > 0]
    if not parts:
        return Polygon()
    return unary_union(parts)


def _sample_ring(ring, segment_m: float) -> list[Point]:
    length = float(ring.length)
    if length <= 0:
        return []
    count = max(1, int(math.ceil(length / segment_m)))
    return [ring.interpolate(index * length / count) for index in range(count)]


def _sample_polygon_boundary(
    geometry: BaseGeometry, segment_m: float
) -> list[Point]:
    points: list[Point] = []
    for polygon in _polygon_parts(geometry):
        points.extend(_sample_ring(polygon.exterior, segment_m))
        for interior in polygon.interiors:
            points.extend(_sample_ring(interior, segment_m))
    return points


def _cell_record(building_id: str, geometry: BaseGeometry) -> dict[str, Any]:
    return {
        "building_id": building_id,
        "geometry_wkt_metric": geometry.wkt,
        "area_m2": float(geometry.area),
        "perimeter_m": float(geometry.length),
        "geometry_valid": bool(geometry.is_valid),
    }


def build_morphological_cells(
    block_geometry: BaseGeometry,
    buildings: list[dict[str, Any]],
    config: TessellationConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Build footprint-sensitive cells from discretized building boundaries."""
    started = time.perf_counter()
    if block_geometry is None or block_geometry.is_empty:
        raise ValueError("block_geometry must be non-empty")
    if not block_geometry.is_valid:
        block_geometry = _polygonal_geometry(make_valid(block_geometry))
    if block_geometry.is_empty or block_geometry.area <= 0:
        raise ValueError("block_geometry must have positive polygonal area")

    failures: list[dict[str, Any]] = []
    valid: list[dict[str, Any]] = []
    sampled_points: list[Point] = []
    point_owners: list[str] = []

    for building in buildings:
        building_id = str(building["building_id"])
        try:
            geometry = building.get("geometry")
            if geometry is None:
                geometry = wkt.loads(building["geometry_wkt_metric"])
        except Exception as error:
            failures.append(
                {
                    "building_id": building_id,
                    "failure_type": "invalid_input_geometry",
                    "message": str(error),
                }
            )
            continue
        if geometry.is_empty:
            failures.append(
                {
                    "building_id": building_id,
                    "failure_type": "invalid_input_geometry",
                    "message": "Input footprint is empty",
                }
            )
            continue
        if not geometry.is_valid:
            geometry = _polygonal_geometry(make_valid(geometry))
        else:
            geometry = _polygonal_geometry(geometry)
        if geometry.is_empty or geometry.area <= 0:
            failures.append(
                {
                    "building_id": building_id,
                    "failure_type": "invalid_input_geometry",
                    "message": "Input footprint has no valid polygonal area",
                }
            )
            continue
        shrunken = _polygonal_geometry(geometry.buffer(-config.shrink_m))
        if shrunken.is_empty or shrunken.area <= 0:
            failures.append(
                {
                    "building_id": building_id,
                    "failure_type": "empty_after_shrink",
                    "message": f"Footprint became empty after {-config.shrink_m:g} m buffer",
                }
            )
            continue
        points = _sample_polygon_boundary(shrunken, config.segment_m)
        if not points:
            failures.append(
                {
                    "building_id": building_id,
                    "failure_type": "voronoi_failure",
                    "message": "No boundary points could be sampled",
                }
            )
            continue
        valid.append({"building_id": building_id, "geometry": geometry})
        sampled_points.extend(points)
        point_owners.extend([building_id] * len(points))

    cells: list[dict[str, Any]] = []
    status = "voronoi"
    if len(valid) == 1:
        status = "single_building_rule" if len(buildings) == 1 else "single_valid_building_rule"
        cells = [_cell_record(valid[0]["building_id"], block_geometry)]
    elif len(valid) >= 2:
        try:
            diagram_edges = voronoi_polygons(
                MultiPoint(sampled_points),
                extend_to=block_geometry.envelope,
                only_edges=True,
            )
            network = unary_union([diagram_edges, block_geometry.boundary])
            coordinates = np.asarray(
                [(point.x, point.y) for point in sampled_points], dtype=float
            )
            nearest_site = cKDTree(coordinates)
            pieces_by_building: dict[str, list[BaseGeometry]] = defaultdict(list)
            for region in polygonize(network):
                representative = region.representative_point()
                if not block_geometry.covers(representative):
                    continue
                clipped = _polygonal_geometry(region.intersection(block_geometry))
                if clipped.is_empty or clipped.area <= 0:
                    continue
                _, site_index = nearest_site.query(
                    [representative.x, representative.y], k=1
                )
                owner = point_owners[int(site_index)]
                pieces_by_building[owner].append(clipped)
            for item in valid:
                building_id = item["building_id"]
                merged = _polygonal_geometry(
                    unary_union(pieces_by_building.get(building_id, []))
                )
                merged = _polygonal_geometry(merged.intersection(block_geometry))
                if not merged.is_valid:
                    merged = _polygonal_geometry(make_valid(merged))
                if merged.is_empty or merged.area <= 0:
                    failures.append(
                        {
                            "building_id": building_id,
                            "failure_type": "empty_after_clip",
                            "message": "Merged Voronoi region is empty after block clipping",
                        }
                    )
                else:
                    cells.append(_cell_record(building_id, merged))
        except Exception as error:
            status = "voronoi_failure"
            cells = []
            existing_failures = {item["building_id"] for item in failures}
            for item in valid:
                if item["building_id"] not in existing_failures:
                    failures.append(
                        {
                            "building_id": item["building_id"],
                            "failure_type": "voronoi_failure",
                            "message": str(error),
                        }
                    )
    elif buildings:
        status = "no_valid_buildings"

    cell_geometries = [wkt.loads(cell["geometry_wkt_metric"]) for cell in cells]
    cell_union = unary_union(cell_geometries) if cell_geometries else Polygon()
    overlap_area = max(
        0.0,
        sum(float(geometry.area) for geometry in cell_geometries)
        - float(cell_union.area),
    )
    outside_area = sum(
        float(geometry.difference(block_geometry).area) for geometry in cell_geometries
    )
    uncovered_area = float(block_geometry.difference(cell_union).area)
    diagnostics = {
        "construction_status": status,
        "config": asdict(config),
        "input_building_count": len(buildings),
        "valid_shrunken_building_count": len(valid),
        "success_cell_count": len(cells),
        "failure_count": len(failures),
        "sample_point_count": len(sampled_points),
        "cell_overlap_area_m2": overlap_area,
        "cell_outside_area_m2": outside_area,
        "uncovered_area_m2": uncovered_area,
        "uncovered_area_ratio": uncovered_area / float(block_geometry.area),
        "operations": [
            "make_valid_if_needed",
            "negative_buffer",
            "boundary_segmentize",
            "point_voronoi_edges",
            "polygonize_with_block_boundary",
            "nearest_site_owner",
            "union_by_building",
            "clip_to_block",
        ],
        "runtime_seconds": time.perf_counter() - started,
    }
    if uncovered_area > max(1e-8, float(block_geometry.area) * 1e-10):
        diagnostics["warnings"] = ["unassigned_region"]
    else:
        diagnostics["warnings"] = []
    return cells, failures, diagnostics


def _record_geometry(record: dict[str, Any]) -> BaseGeometry:
    geometry = record.get("geometry")
    if geometry is None:
        geometry = wkt.loads(record["geometry_wkt_metric"])
    if not geometry.is_valid:
        geometry = make_valid(geometry)
    geometry = _polygonal_geometry(geometry)
    if geometry.is_empty or geometry.area <= 0:
        raise ValueError("record has no positive polygonal geometry")
    return geometry


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and math.isfinite(
        float(value)
    )


def _positive_number(value: Any) -> bool:
    return _finite_number(value) and float(value) > 0


def _axial_difference_deg(angle_a: float, angle_b: float) -> float:
    return abs((float(angle_a) - float(angle_b) + 90.0) % 180.0 - 90.0)


def _representative_xy(
    record: dict[str, Any], geometry: BaseGeometry
) -> tuple[float, float]:
    x = record.get("representative_point_x_m")
    y = record.get("representative_point_y_m")
    if _finite_number(x) and _finite_number(y):
        return float(x), float(y)
    point = geometry.centroid
    return float(point.x), float(point.y)


def _axis_unit(angle_deg: float) -> np.ndarray:
    angle_rad = math.radians(float(angle_deg) % 180.0)
    return np.asarray([math.cos(angle_rad), math.sin(angle_rad)], dtype=float)


def _projection_interval(
    geometry: BaseGeometry, axis: np.ndarray
) -> tuple[float, float]:
    rectangle = geometry.minimum_rotated_rectangle
    coordinates = np.asarray(rectangle.exterior.coords[:-1], dtype=float)
    projected = coordinates @ axis
    return float(projected.min()), float(projected.max())


def _long_side_relations(
    geometry_i: BaseGeometry,
    geometry_j: BaseGeometry,
    orientation_i: float,
    orientation_j: float,
) -> tuple[float, float]:
    axis_i = _axis_unit(orientation_i)
    axis_j = _axis_unit(orientation_j)
    if float(axis_i @ axis_j) < 0:
        axis_j = -axis_j
    mean_axis = axis_i + axis_j
    norm = float(np.linalg.norm(mean_axis))
    if norm <= 1e-12:
        mean_axis = axis_i
    else:
        mean_axis = mean_axis / norm
    normal_axis = np.asarray([-mean_axis[1], mean_axis[0]], dtype=float)

    along_i = _projection_interval(geometry_i, mean_axis)
    along_j = _projection_interval(geometry_j, mean_axis)
    overlap = max(0.0, min(along_i[1], along_j[1]) - max(along_i[0], along_j[0]))
    shorter_span = min(along_i[1] - along_i[0], along_j[1] - along_j[0])
    if shorter_span <= 0:
        raise ValueError("oriented rectangle has zero projected long-side span")
    overlap_ratio = min(1.0, max(0.0, overlap / shorter_span))

    across_i = _projection_interval(geometry_i, normal_axis)
    across_j = _projection_interval(geometry_j, normal_axis)
    side_offset = min(
        abs(value_i - value_j)
        for value_i in across_i
        for value_j in across_j
    )
    local_scale = math.sqrt((float(geometry_i.area) + float(geometry_j.area)) / 2.0)
    if local_scale <= 0:
        raise ValueError("local building scale is zero")
    return float(overlap_ratio), float(side_offset / local_scale)


def build_cell_adjacency(
    cells: list[dict[str, Any]],
    buildings: list[dict[str, Any]],
    block_area_m2: float,
    config: TessellationConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Create stable undirected edges from shared cell line boundaries."""
    started = time.perf_counter()
    if not _positive_number(block_area_m2):
        raise ValueError("block_area_m2 must be finite and positive")
    block_scale = math.sqrt(float(block_area_m2))
    building_by_id = {str(item["building_id"]): item for item in buildings}
    geometry_by_id = {
        building_id: _record_geometry(record)
        for building_id, record in building_by_id.items()
    }

    ordered_cells = sorted(cells, key=lambda item: str(item["building_id"]))
    cell_ids = [str(item["building_id"]) for item in ordered_cells]
    cell_geometries = [_record_geometry(item) for item in ordered_cells]
    tree = __import__("shapely").STRtree(cell_geometries)
    edges: list[dict[str, Any]] = []
    point_contacts_excluded = 0

    for index_i, cell_i in enumerate(cell_geometries):
        for candidate in tree.query(cell_i, predicate="intersects"):
            index_j = int(candidate)
            if index_j <= index_i:
                continue
            cell_j = cell_geometries[index_j]
            shared_length = float(cell_i.boundary.intersection(cell_j.boundary).length)
            if shared_length <= config.min_shared_boundary_m:
                point_contacts_excluded += 1
                continue

            source = cell_ids[index_i]
            target = cell_ids[index_j]
            if source not in geometry_by_id or target not in geometry_by_id:
                continue
            record_i = building_by_id[source]
            record_j = building_by_id[target]
            geometry_i = geometry_by_id[source]
            geometry_j = geometry_by_id[target]
            point_i = _representative_xy(record_i, geometry_i)
            point_j = _representative_xy(record_j, geometry_j)
            delta_x = point_j[0] - point_i[0]
            delta_y = point_j[1] - point_i[1]
            centroid_distance = math.hypot(delta_x, delta_y)
            footprint_gap = float(geometry_i.distance(geometry_j))

            perimeter_i = float(cell_i.length)
            perimeter_j = float(cell_j.length)
            minimum_perimeter = min(perimeter_i, perimeter_j)
            mean_perimeter = (perimeter_i + perimeter_j) / 2.0
            missing: dict[str, bool] = {}
            missing_reasons: dict[str, str | None] = {}

            def set_optional(field: str, value: float | None, reason: str | None = None):
                missing[field] = value is None
                missing_reasons[field] = reason if value is None else None
                return value

            orientation_i = record_i.get("orientation_deg")
            orientation_j = record_j.get("orientation_deg")
            orientation_valid = _finite_number(orientation_i) and _finite_number(orientation_j)
            if orientation_valid:
                orientation_diff = _axial_difference_deg(
                    float(orientation_i), float(orientation_j)
                )
                orientation_similarity = (
                    1.0 + math.cos(2.0 * math.radians(orientation_diff))
                ) / 2.0
                bearing = math.degrees(math.atan2(delta_y, delta_x)) % 180.0
                alignment_i = _axial_difference_deg(float(orientation_i), bearing)
                alignment_j = _axial_difference_deg(float(orientation_j), bearing)
                try:
                    overlap_ratio, side_offset_norm = _long_side_relations(
                        geometry_i,
                        geometry_j,
                        float(orientation_i),
                        float(orientation_j),
                    )
                    long_side_reason = None
                except Exception as error:
                    overlap_ratio = None
                    side_offset_norm = None
                    long_side_reason = f"long_side_calculation_failed: {error}"
            else:
                orientation_diff = None
                orientation_similarity = None
                alignment_i = None
                alignment_j = None
                overlap_ratio = None
                side_offset_norm = None
                long_side_reason = "missing_or_invalid_orientation"

            area_i = float(geometry_i.area)
            area_j = float(geometry_j.area)
            log_area_ratio = (
                abs(math.log(area_i / area_j))
                if area_i > 0 and area_j > 0
                else None
            )
            height_i = record_i.get("height_m")
            height_j = record_j.get("height_m")
            heights_valid = _positive_number(height_i) and _positive_number(height_j)
            if heights_valid:
                height_i_value = float(height_i)
                height_j_value = float(height_j)
                height_diff = abs(height_i_value - height_j_value)
                log_height_ratio = abs(math.log(height_i_value / height_j_value))
                volume_i = area_i * height_i_value
                volume_j = area_j * height_j_value
                volume_ratio = max(volume_i, volume_j) / min(volume_i, volume_j)
                height_reason = None
            else:
                height_diff = None
                log_height_ratio = None
                volume_ratio = None
                height_reason = "missing_or_nonpositive_height"

            edge = {
                "edge_id": f"celladj|{source}|{target}",
                "source": source,
                "target": target,
                "shared_cell_boundary_m": shared_length,
                "shared_boundary_ratio_min": shared_length / minimum_perimeter,
                "shared_boundary_ratio_mean": shared_length / mean_perimeter,
                "footprint_gap_m": footprint_gap,
                "footprint_gap_block_norm": footprint_gap / block_scale,
                "centroid_distance_m": centroid_distance,
                "centroid_distance_block_norm": centroid_distance / block_scale,
                "orientation_diff_deg": set_optional(
                    "orientation_diff_deg", orientation_diff, "missing_or_invalid_orientation"
                ),
                "orientation_similarity": set_optional(
                    "orientation_similarity",
                    orientation_similarity,
                    "missing_or_invalid_orientation",
                ),
                "long_side_overlap_ratio": set_optional(
                    "long_side_overlap_ratio", overlap_ratio, long_side_reason
                ),
                "long_side_offset_norm": set_optional(
                    "long_side_offset_norm", side_offset_norm, long_side_reason
                ),
                "connection_axis_alignment_i": set_optional(
                    "connection_axis_alignment_i",
                    alignment_i,
                    "missing_or_invalid_orientation",
                ),
                "connection_axis_alignment_j": set_optional(
                    "connection_axis_alignment_j",
                    alignment_j,
                    "missing_or_invalid_orientation",
                ),
                "log_area_ratio_abs": set_optional(
                    "log_area_ratio_abs", log_area_ratio, "missing_or_nonpositive_area"
                ),
                "height_diff_m": set_optional(
                    "height_diff_m", height_diff, height_reason
                ),
                "log_height_ratio_abs": set_optional(
                    "log_height_ratio_abs", log_height_ratio, height_reason
                ),
                "volume_ratio": set_optional("volume_ratio", volume_ratio, height_reason),
                "missing": missing,
                "missing_reasons": missing_reasons,
            }
            edges.append(edge)

    edges.sort(key=lambda item: item["edge_id"])
    diagnostics = {
        "cell_count": len(cells),
        "edge_count": len(edges),
        "point_contacts_excluded": point_contacts_excluded,
        "min_shared_boundary_m": config.min_shared_boundary_m,
        "runtime_seconds": time.perf_counter() - started,
    }
    return edges, diagnostics


def validate_tessellation_graph(
    graph: dict[str, Any],
) -> dict[str, Any]:
    """Return machine-readable graph invariant violations."""
    issues: list[dict[str, str]] = []
    buildings = graph.get("nodes", [])
    cells = graph.get("cells", [])
    edges = graph.get("edges", [])
    node_ids = {str(item["building_id"]) for item in buildings}
    cell_ids = [str(item["building_id"]) for item in cells]
    if len(cell_ids) != len(set(cell_ids)):
        issues.append({"issue_type": "duplicate_cell_id", "message": "Cell IDs are not unique"})
    for cell_id in cell_ids:
        if cell_id not in node_ids:
            issues.append(
                {"issue_type": "dangling_cell", "message": f"Cell {cell_id} has no node"}
            )

    seen_edges: set[str] = set()
    nonnegative_fields = (
        "shared_cell_boundary_m",
        "shared_boundary_ratio_min",
        "shared_boundary_ratio_mean",
        "footprint_gap_m",
        "footprint_gap_block_norm",
        "centroid_distance_m",
        "centroid_distance_block_norm",
        "long_side_overlap_ratio",
        "long_side_offset_norm",
        "log_area_ratio_abs",
        "height_diff_m",
        "log_height_ratio_abs",
        "volume_ratio",
    )
    bounded_fields = {
        "shared_boundary_ratio_min": (0.0, 1.0),
        "shared_boundary_ratio_mean": (0.0, 1.0),
        "orientation_diff_deg": (0.0, 90.0),
        "orientation_similarity": (0.0, 1.0),
        "long_side_overlap_ratio": (0.0, 1.0),
        "connection_axis_alignment_i": (0.0, 90.0),
        "connection_axis_alignment_j": (0.0, 90.0),
    }
    for edge in edges:
        edge_id = str(edge.get("edge_id", ""))

        def add(issue_type: str, message: str) -> None:
            issues.append(
                {"issue_type": issue_type, "edge_id": edge_id, "message": message}
            )

        source = str(edge.get("source", ""))
        target = str(edge.get("target", ""))
        if edge_id in seen_edges:
            add("duplicate_edge_id", "Edge ID is duplicated")
        seen_edges.add(edge_id)
        if source == target:
            add("self_loop", "Edge endpoints are identical")
        if source not in node_ids or target not in node_ids:
            add("dangling_edge", "Edge endpoint is absent from nodes")
        if source >= target:
            add("unstable_endpoint_order", "Undirected endpoints are not sorted")
        if edge_id != f"celladj|{source}|{target}":
            add("unstable_edge_id", "Edge ID does not match sorted endpoints")
        for field in nonnegative_fields:
            value = edge.get(field)
            if value is not None and (not _finite_number(value) or float(value) < 0):
                add("invalid_nonnegative_value", f"{field} must be finite and non-negative")
        for field, (minimum, maximum) in bounded_fields.items():
            value = edge.get(field)
            if value is not None and not (minimum - 1e-9 <= float(value) <= maximum + 1e-9):
                add("out_of_range", f"{field} must be in [{minimum}, {maximum}]")
        value = edge.get("volume_ratio")
        if value is not None and float(value) < 1.0 - 1e-9:
            add("out_of_range", "volume_ratio must be at least 1")
        for field in ("height_diff_m", "log_height_ratio_abs", "volume_ratio"):
            value = edge.get(field)
            missing = edge.get("missing", {}).get(field)
            reason = edge.get("missing_reasons", {}).get(field)
            if value is None and (missing is not True or not reason):
                add("missing_flag_mismatch", f"{field} null value lacks flag and reason")
            if value is not None and missing is not False:
                add("missing_flag_mismatch", f"{field} value is marked missing")
    issue_counts = Counter(item["issue_type"] for item in issues)
    return {
        "passed": not issues,
        "issue_count": len(issues),
        "issue_counts": dict(sorted(issue_counts.items())),
        "issues": issues,
    }
