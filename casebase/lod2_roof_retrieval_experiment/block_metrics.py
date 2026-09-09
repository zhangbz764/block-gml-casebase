"""Building- and block-level metrics for the Luxembourg LOD2 experiment."""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from shapely.geometry import Polygon

from database import BuildingGeometryRecord
from roof_geometry import (
    ground_elevation,
    measure_polygon_3d,
    roof_relief_ratio,
    triangulate_polygon_3d,
)


M1_CANDIDATE_FEATURES = [
    "log_block_area",
    "compactness",
    "elongation",
    "bcr",
    "volume_density",
    "mean_h_g1",
    "h_g1_cv",
    "log_building_count",
    "mean_footprint_area",
    "footprint_area_cv",
    "orientation_entropy",
    "normalized_nn_distance",
]

ROOF_FEATURES = [
    "roof_mean_slope",
    "roof_relief_ratio_median",
    "building_roof_slope_sd",
]

BUILDING_METRIC_COLUMNS = [
    "building_id",
    "block_id",
    "footprint_area",
    "centroid_x",
    "centroid_y",
    "orientation_deg",
    "h_g1",
    "roof_area_3d",
    "roof_projected_area",
    "roof_mean_slope",
    "roof_relief_ratio",
    "roof_surface_count",
    "ground_surface_count",
    "valid_triangle_count",
    "is_usable",
    "quality_codes",
]

BLOCK_METRIC_COLUMNS = [
    "block_id",
    "area_m2",
    "compactness",
    "elongation",
    "bcr",
    "volume_density",
    "mean_h_g1",
    "h_g1_cv",
    "building_count",
    "log_block_area",
    "log_building_count",
    "mean_footprint_area",
    "footprint_area_cv",
    "orientation_entropy",
    "normalized_nn_distance",
    "roof_mean_slope",
    "roof_relief_ratio_median",
    "building_roof_slope_sd",
    "usable_footprint_ratio",
    "is_usable",
    "quality_codes",
]


def _orientation_deg(polygon: Polygon) -> float:
    rectangle = polygon.minimum_rotated_rectangle
    coordinates = list(rectangle.exterior.coords)
    edges = []
    for start, end in zip(coordinates[:-1], coordinates[1:]):
        dx, dy = end[0] - start[0], end[1] - start[1]
        edges.append((math.hypot(dx, dy), math.degrees(math.atan2(dy, dx)) % 180.0))
    if not edges or max(edges)[0] <= 0:
        return float("nan")
    return float(max(edges, key=lambda item: item[0])[1])


def _semicolon_codes(codes: Iterable[str]) -> str:
    return ";".join(dict.fromkeys(code for code in codes if code))


def compute_building_metric_row(record: BuildingGeometryRecord) -> dict:
    """Measure one building and retain pragmatic, auditable quality flags."""

    codes: list[str] = list(record.source_quality_codes)
    footprint_area = float("nan")
    centroid_x = float("nan")
    centroid_y = float("nan")
    orientation = float("nan")
    if record.footprint is None or record.footprint.is_empty or record.footprint.area <= 0:
        codes.append("invalid_footprint")
    else:
        footprint_area = float(record.footprint.area)
        centroid_x = float(record.footprint.centroid.x)
        centroid_y = float(record.footprint.centroid.y)
        orientation = _orientation_deg(record.footprint)

    if not record.roof_polygons:
        codes.append("missing_roof")
    if not record.ground_polygons:
        codes.append("missing_ground")

    valid_roofs = []
    roof_measurements = []
    valid_triangle_count = 0
    invalid_roofs = 0
    for roof in record.roof_polygons:
        try:
            measurement = measure_polygon_3d(roof)
            triangle_count = len(triangulate_polygon_3d(roof))
        except (TypeError, ValueError):
            invalid_roofs += 1
            continue
        valid_roofs.append(roof)
        roof_measurements.append(measurement)
        valid_triangle_count += triangle_count
    if invalid_roofs:
        codes.append(f"invalid_roof_surfaces:{invalid_roofs}")
    if record.roof_polygons and not valid_roofs:
        codes.append("no_valid_roof_surface")

    roof_area_3d = sum(item.area_3d for item in roof_measurements)
    roof_projected_area = sum(item.projected_area for item in roof_measurements)
    roof_mean_slope = (
        sum(item.mean_slope_deg * item.area_3d for item in roof_measurements)
        / roof_area_3d
        if roof_area_3d > 0
        else float("nan")
    )

    ground_z = float("nan")
    if record.ground_polygons:
        try:
            ground_z = ground_elevation(record.ground_polygons)
        except (TypeError, ValueError):
            codes.append("invalid_ground_surface")

    h_g1 = float("nan")
    if roof_projected_area > 0 and np.isfinite(ground_z):
        mean_roof_z = sum(
            item.mean_z * item.projected_area for item in roof_measurements
        ) / roof_projected_area
        h_g1 = float(mean_roof_z - ground_z)
        if h_g1 <= 0:
            codes.append("non_positive_height")

    relief = float("nan")
    if valid_roofs and np.isfinite(ground_z):
        try:
            relief = roof_relief_ratio(valid_roofs, record.ground_polygons)
        except (TypeError, ValueError):
            codes.append("invalid_roof_relief")

    hard_valid = (
        np.isfinite(footprint_area)
        and footprint_area > 0
        and roof_area_3d > 0
        and roof_projected_area > 0
        and np.isfinite(h_g1)
        and h_g1 > 0
        and np.isfinite(roof_mean_slope)
        and 0 <= roof_mean_slope <= 90
        and np.isfinite(relief)
        and 0 <= relief <= 1
    )
    if not hard_valid and not codes:
        codes.append("invalid_required_metric")

    return {
        "building_id": record.building_id,
        "block_id": record.block_id,
        "footprint_area": footprint_area,
        "centroid_x": centroid_x,
        "centroid_y": centroid_y,
        "orientation_deg": orientation,
        "h_g1": h_g1,
        "roof_area_3d": float(roof_area_3d),
        "roof_projected_area": float(roof_projected_area),
        "roof_mean_slope": float(roof_mean_slope),
        "roof_relief_ratio": float(relief),
        "roof_surface_count": len(record.roof_polygons),
        "ground_surface_count": len(record.ground_polygons),
        "valid_triangle_count": valid_triangle_count,
        "is_usable": bool(hard_valid),
        "quality_codes": _semicolon_codes(codes),
    }


def compute_building_metrics(records: Iterable[BuildingGeometryRecord]) -> pd.DataFrame:
    """Compute one auditable metric row per streamed building record."""

    return pd.DataFrame(
        (compute_building_metric_row(record) for record in records),
        columns=BUILDING_METRIC_COLUMNS,
    )


def _coefficient_of_variation(values: np.ndarray) -> float:
    mean = float(np.mean(values))
    return float(np.std(values, ddof=0) / mean) if mean > 0 else float("nan")


def _block_elongation(block: Polygon) -> float:
    rectangle = block.minimum_rotated_rectangle
    coordinates = list(rectangle.exterior.coords)
    lengths = [
        math.hypot(end[0] - start[0], end[1] - start[1])
        for start, end in zip(coordinates[:-1], coordinates[1:])
    ]
    positive = [value for value in lengths if value > 0]
    return float(min(positive) / max(positive)) if positive else float("nan")


def _orientation_entropy(values: np.ndarray, bins: int = 18) -> float:
    if values.size <= 1:
        return 0.0
    counts, _ = np.histogram(values % 180.0, bins=bins, range=(0.0, 180.0))
    probabilities = counts[counts > 0] / counts.sum()
    entropy = -float(np.sum(probabilities * np.log(probabilities)))
    return entropy / math.log(bins)


def nearest_neighbour_distances(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Return each point's Euclidean distance to its nearest other point."""

    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    if xs.shape != ys.shape or xs.ndim != 1:
        raise ValueError("xs and ys must be one-dimensional arrays of equal length")
    if xs.size <= 1:
        return np.zeros(xs.size, dtype=float)
    points = np.column_stack((xs, ys))
    if not np.isfinite(points).all():
        raise ValueError("nearest-neighbour coordinates must be finite")
    distances, _ = cKDTree(points).query(points, k=2)
    return distances[:, 1]


def _normalized_nearest_neighbour_distance(
    xs: np.ndarray, ys: np.ndarray, block_area: float
) -> float:
    if xs.size <= 1:
        return 0.0
    distances = nearest_neighbour_distances(xs, ys)
    return float(np.mean(distances) / math.sqrt(block_area))


def compute_block_metrics(
    blocks: pd.DataFrame,
    buildings: pd.DataFrame,
    coverage_threshold: float = 0.80,
) -> pd.DataFrame:
    """Aggregate a reusable LOD1 baseline and three roof descriptors per block."""

    rows = []
    building_groups = {key: group for key, group in buildings.groupby("block_id")}
    for block in blocks.itertuples(index=False):
        group = building_groups.get(block.block_id, buildings.iloc[0:0])
        footprint_all = pd.to_numeric(group.get("footprint_area"), errors="coerce")
        all_area = float(footprint_all[np.isfinite(footprint_all)].sum())
        usable = group[group.get("is_usable", False).astype(bool)].copy()
        usable_area = float(usable["footprint_area"].sum()) if not usable.empty else 0.0
        coverage = usable_area / all_area if all_area > 0 else 0.0
        codes = []
        if coverage < coverage_threshold:
            codes.append("coverage_below_80pct")
        if usable.empty:
            codes.append("no_usable_buildings")

        area = float(block.area_m2)
        perimeter = float(block.perimeter_m)
        footprint = usable["footprint_area"].to_numpy(dtype=float)
        heights = usable["h_g1"].to_numpy(dtype=float)
        slopes = usable["roof_mean_slope"].to_numpy(dtype=float)
        roof_areas = usable["roof_area_3d"].to_numpy(dtype=float)
        orientations = usable["orientation_deg"].to_numpy(dtype=float)
        building_count = len(usable)

        roof_mean = (
            float(np.average(slopes, weights=roof_areas))
            if building_count and roof_areas.sum() > 0
            else float("nan")
        )
        metrics = {
            "block_id": block.block_id,
            "area_m2": area,
            "compactness": 4.0 * math.pi * area / perimeter**2
            if perimeter > 0
            else float("nan"),
            "elongation": _block_elongation(block.geometry),
            "bcr": usable_area / area if area > 0 else float("nan"),
            "volume_density": float(np.sum(footprint * heights) / area)
            if area > 0 and building_count
            else float("nan"),
            "mean_h_g1": float(np.mean(heights)) if building_count else float("nan"),
            "h_g1_cv": _coefficient_of_variation(heights)
            if building_count
            else float("nan"),
            "building_count": building_count,
            "log_block_area": math.log(area) if area > 0 else float("nan"),
            "log_building_count": math.log(building_count)
            if building_count > 0
            else float("nan"),
            "mean_footprint_area": float(np.mean(footprint))
            if building_count
            else float("nan"),
            "footprint_area_cv": _coefficient_of_variation(footprint)
            if building_count
            else float("nan"),
            "orientation_entropy": _orientation_entropy(orientations[np.isfinite(orientations)]),
            "normalized_nn_distance": _normalized_nearest_neighbour_distance(
                usable["centroid_x"].to_numpy(dtype=float),
                usable["centroid_y"].to_numpy(dtype=float),
                area,
            )
            if area > 0 and building_count
            else float("nan"),
            "roof_mean_slope": roof_mean,
            "roof_relief_ratio_median": float(np.median(usable["roof_relief_ratio"]))
            if building_count
            else float("nan"),
            "building_roof_slope_sd": float(np.std(slopes, ddof=0))
            if building_count
            else float("nan"),
            "usable_footprint_ratio": coverage,
        }
        required = M1_CANDIDATE_FEATURES + ROOF_FEATURES
        if any(not np.isfinite(metrics[name]) for name in required):
            codes.append("non_finite_required_metric")
        metrics["is_usable"] = not codes
        metrics["quality_codes"] = _semicolon_codes(codes)
        rows.append(metrics)
    return pd.DataFrame(rows, columns=BLOCK_METRIC_COLUMNS)


def select_final_m1_features(
    frame: pd.DataFrame,
    threshold: float = 0.90,
    candidate_features: Sequence[str] = M1_CANDIDATE_FEATURES,
) -> tuple[list[str], pd.DataFrame]:
    """Prune highly correlated M1 features using a fixed, auditable rule."""

    ordered = list(candidate_features)
    correlations = frame[ordered].corr(method="spearman")
    missing = frame[ordered].isna().sum()
    retained = ordered.copy()
    decisions = []
    for left_index, left in enumerate(ordered):
        if left not in retained:
            continue
        for right in ordered[left_index + 1 :]:
            if right not in retained:
                continue
            correlation = correlations.loc[left, right]
            if not np.isfinite(correlation) or abs(correlation) <= threshold:
                continue
            if missing[left] > missing[right]:
                dropped, reason = left, "more_missing_values"
            elif missing[right] > missing[left]:
                dropped, reason = right, "more_missing_values"
            else:
                dropped, reason = right, "same_missingness_drop_later"
            retained.remove(dropped)
            decisions.append(
                {
                    "feature_a": left,
                    "feature_b": right,
                    "spearman_correlation": float(correlation),
                    "dropped_feature": dropped,
                    "reason": reason,
                }
            )
            if dropped == left:
                break
    return retained, pd.DataFrame(decisions)
