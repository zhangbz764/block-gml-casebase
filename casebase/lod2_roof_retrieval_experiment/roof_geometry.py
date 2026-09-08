"""Three-dimensional roof measurements in a local metric coordinate system."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np
from shapely.geometry import Polygon
from shapely.ops import triangulate


_EPSILON = 1e-9


@dataclass(frozen=True)
class FaceMetrics:
    """Area and elevation measurements aggregated over one polygonal face."""

    projected_area: float
    area_3d: float
    mean_slope_deg: float
    mean_z: float


def _coordinate_lookup(polygon: Polygon) -> dict[tuple[float, float], float]:
    lookup: dict[tuple[float, float], float] = {}
    rings = [polygon.exterior, *polygon.interiors]
    for ring in rings:
        for coordinate in ring.coords:
            if len(coordinate) < 3 or not np.isfinite(coordinate[2]):
                raise ValueError("Polygon must contain finite Z coordinates")
            lookup[(round(float(coordinate[0]), 9), round(float(coordinate[1]), 9))] = float(
                coordinate[2]
            )
    return lookup


def triangulate_polygon_3d(polygon: Polygon) -> list[np.ndarray]:
    """Triangulate a PolygonZ in XY while restoring each source vertex's Z."""

    if polygon.is_empty or not polygon.has_z:
        raise ValueError("Expected a non-empty PolygonZ")
    z_lookup = _coordinate_lookup(polygon)
    triangles: list[np.ndarray] = []
    for triangle_2d in triangulate(polygon):
        if not polygon.covers(triangle_2d.representative_point()):
            continue
        coordinates = list(triangle_2d.exterior.coords)[:3]
        points = []
        for x, y, *_ in coordinates:
            key = (round(float(x), 9), round(float(y), 9))
            if key not in z_lookup:
                raise ValueError("Triangulation introduced an unmapped XY vertex")
            points.append((float(x), float(y), z_lookup[key]))
        triangles.append(np.asarray(points, dtype=float))
    if not triangles:
        raise ValueError("Polygon produced no usable triangles")
    return triangles


def measure_polygon_3d(polygon: Polygon) -> FaceMetrics:
    """Measure a roof face using 3D area and horizontal-projection weights."""

    projected_area = 0.0
    area_3d = 0.0
    slope_area_sum = 0.0
    z_projected_area_sum = 0.0
    for triangle in triangulate_polygon_3d(polygon):
        edge_a = triangle[1] - triangle[0]
        edge_b = triangle[2] - triangle[0]
        normal = np.cross(edge_a, edge_b)
        normal_length = float(np.linalg.norm(normal))
        triangle_area_3d = 0.5 * normal_length
        triangle_projected_area = 0.5 * abs(float(normal[2]))
        if triangle_area_3d <= _EPSILON:
            continue
        cosine_from_vertical = min(1.0, abs(float(normal[2])) / normal_length)
        slope_deg = math.degrees(math.acos(cosine_from_vertical))
        area_3d += triangle_area_3d
        projected_area += triangle_projected_area
        slope_area_sum += slope_deg * triangle_area_3d
        z_projected_area_sum += float(np.mean(triangle[:, 2])) * triangle_projected_area
    if area_3d <= _EPSILON or projected_area <= _EPSILON:
        raise ValueError("Polygon has no non-degenerate roof area")
    return FaceMetrics(
        projected_area=projected_area,
        area_3d=area_3d,
        mean_slope_deg=slope_area_sum / area_3d,
        mean_z=z_projected_area_sum / projected_area,
    )


def _all_z_values(polygons: Iterable[Polygon]) -> list[float]:
    values: list[float] = []
    for polygon in polygons:
        rings = [polygon.exterior, *polygon.interiors]
        for ring in rings:
            values.extend(float(coordinate[2]) for coordinate in ring.coords)
    return values


def ground_elevation(ground_polygons: Iterable[Polygon]) -> float:
    """Return the median Z over all ground-surface vertices."""

    values = _all_z_values(ground_polygons)
    if not values or not np.all(np.isfinite(values)):
        raise ValueError("Ground surfaces do not contain finite Z coordinates")
    return float(np.median(values))


def g1_height(
    roof_polygons: Iterable[Polygon], ground_polygons: Iterable[Polygon]
) -> float:
    """Compute projected-area-weighted mean roof elevation above ground."""

    roof_measurements = [measure_polygon_3d(polygon) for polygon in roof_polygons]
    if not roof_measurements:
        raise ValueError("At least one roof surface is required")
    total_projected_area = sum(item.projected_area for item in roof_measurements)
    mean_roof_z = sum(
        item.mean_z * item.projected_area for item in roof_measurements
    ) / total_projected_area
    height = mean_roof_z - ground_elevation(ground_polygons)
    if not np.isfinite(height) or height <= 0:
        raise ValueError("G1 height must be finite and positive")
    return float(height)


def roof_relief_ratio(
    roof_polygons: Iterable[Polygon], ground_polygons: Iterable[Polygon]
) -> float:
    """Return roof relief divided by total geometry-derived building height."""

    roof_z = _all_z_values(roof_polygons)
    if not roof_z or not np.all(np.isfinite(roof_z)):
        raise ValueError("Roof surfaces do not contain finite Z coordinates")
    ground_z = ground_elevation(ground_polygons)
    denominator = max(roof_z) - ground_z
    if denominator <= 0:
        return float("nan")
    return float((max(roof_z) - min(roof_z)) / denominator)
