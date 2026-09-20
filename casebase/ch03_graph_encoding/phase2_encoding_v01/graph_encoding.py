"""Model-neutral block graph encoding primitives for Chapter 3 phase 2."""

from __future__ import annotations

import json
import math
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import Delaunay, QhullError
from shapely import wkt
from shapely.geometry.base import BaseGeometry


ENCODING_VERSION = "0.1"


def _finite_or_none(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _orientation(geometry: BaseGeometry) -> tuple[float | None, bool]:
    """Return an axial long-edge orientation in degrees and a stability flag."""
    if geometry.is_empty or not geometry.is_valid or geometry.area <= 0:
        return None, False
    rectangle = geometry.minimum_rotated_rectangle
    coordinates = list(rectangle.exterior.coords)[:4]
    edges: list[tuple[float, float, float]] = []
    for first, second in zip(coordinates, coordinates[1:] + coordinates[:1]):
        dx = second[0] - first[0]
        dy = second[1] - first[1]
        edges.append((math.hypot(dx, dy), dx, dy))
    edges.sort(key=lambda item: item[0], reverse=True)
    if not edges or edges[0][0] <= 0:
        return None, False
    angle = math.degrees(math.atan2(edges[0][2], edges[0][1])) % 180.0
    stable = edges[-1][0] > 0 and edges[0][0] / edges[-1][0] > 1.05
    return angle, stable


def derive_node(
    record: dict[str, Any],
    *,
    block_geometry: BaseGeometry,
    origin_x_m: float,
    origin_y_m: float,
    scale_m: float,
) -> dict[str, Any]:
    """Derive one traceable node record from a projected footprint."""
    geometry = record["geometry"]
    if geometry is None or geometry.is_empty or not geometry.is_valid:
        raise ValueError(f"Invalid footprint for building {record.get('building_id')}")
    if not math.isfinite(scale_m) or scale_m <= 0:
        raise ValueError("scale_m must be finite and greater than zero")

    area = float(geometry.area)
    perimeter = float(geometry.length)
    centroid = geometry.centroid
    rectangle_area = float(geometry.minimum_rotated_rectangle.area)
    hull_area = float(geometry.convex_hull.area)
    orientation_deg, orientation_stable = _orientation(geometry)

    compactness = 4.0 * math.pi * area / (perimeter * perimeter) if perimeter > 0 else None
    rectangularity = area / rectangle_area if rectangle_area > 0 else None
    convexity = area / hull_area if hull_area > 0 else None
    theta = math.radians(orientation_deg) if orientation_deg is not None else None

    height = _finite_or_none(record.get("height"))
    source_area = _finite_or_none(record.get("area"))
    source_perimeter = _finite_or_none(record.get("perimeter"))
    local_x = float(centroid.x) - float(origin_x_m)
    local_y = float(centroid.y) - float(origin_y_m)
    boundary_distance = float(centroid.distance(block_geometry.boundary))

    return {
        "building_id": str(record["building_id"]),
        "citygml_id": record.get("citygml_id"),
        "height_m": height,
        "source_area_m2": source_area,
        "source_perimeter_m": source_perimeter,
        "footprint_area_m2": area,
        "footprint_perimeter_m": perimeter,
        "footprint_compactness": compactness,
        "footprint_rectangularity": rectangularity,
        "footprint_convexity": convexity,
        "orientation_deg": orientation_deg,
        "orientation_sin_2theta": math.sin(2.0 * theta) if theta is not None else None,
        "orientation_cos_2theta": math.cos(2.0 * theta) if theta is not None else None,
        "representative_point_x_m": float(centroid.x),
        "representative_point_y_m": float(centroid.y),
        "local_x_m": local_x,
        "local_y_m": local_y,
        "local_x_normalized": local_x / scale_m,
        "local_y_normalized": local_y / scale_m,
        "boundary_distance_m": boundary_distance,
        "boundary_distance_normalized": boundary_distance / scale_m,
        "geometry_wkt_metric": geometry.wkt,
        "source_index": {
            "building_id": str(record["building_id"]),
            "geometry_column": "geom_2d",
        },
        "missing": {
            "citygml_id": record.get("citygml_id") is None,
            "height_m": height is None,
            "source_area_m2": source_area is None,
            "source_perimeter_m": source_perimeter is None,
            "orientation": orientation_deg is None,
        },
        "quality": {
            "footprint_valid": bool(geometry.is_valid),
            "centroid_covered_by_block": bool(block_geometry.covers(centroid)),
            "footprint_covered_by_block": bool(block_geometry.covers(geometry)),
            "orientation_stable": orientation_stable,
        },
    }


def _edge_record(
    first: dict[str, Any], second: dict[str, Any], scale_m: float
) -> dict[str, Any]:
    if first["building_id"] > second["building_id"]:
        first, second = second, first
    dx = second["representative_point_x_m"] - first["representative_point_x_m"]
    dy = second["representative_point_y_m"] - first["representative_point_y_m"]
    distance = math.hypot(dx, dy)
    first_geometry = wkt.loads(first["geometry_wkt_metric"])
    second_geometry = wkt.loads(second["geometry_wkt_metric"])

    first_angle = first.get("orientation_deg")
    second_angle = second.get("orientation_deg")
    orientation_difference = None
    if first_angle is not None and second_angle is not None:
        raw_difference = abs(float(first_angle) - float(second_angle)) % 180.0
        orientation_difference = min(raw_difference, 180.0 - raw_difference)

    first_height = first.get("height_m")
    second_height = second.get("height_m")
    height_difference = None
    height_ratio = None
    if first_height is not None and second_height is not None:
        height_difference = abs(float(second_height) - float(first_height))
        if first_height > 0 and second_height > 0:
            height_ratio = max(first_height, second_height) / min(
                first_height, second_height
            )

    first_area = first.get("footprint_area_m2")
    second_area = second.get("footprint_area_m2")
    area_ratio = None
    if first_area is not None and second_area is not None and first_area > 0 and second_area > 0:
        area_ratio = max(first_area, second_area) / min(first_area, second_area)

    return {
        "source": first["building_id"],
        "target": second["building_id"],
        "relative_dx_m": dx,
        "relative_dy_m": dy,
        "relative_dx_normalized": dx / scale_m,
        "relative_dy_normalized": dy / scale_m,
        "centroid_distance_m": distance,
        "centroid_distance_normalized": distance / scale_m,
        "footprint_distance_m": float(first_geometry.distance(second_geometry)),
        "bearing_cos": dx / distance if distance > 0 else None,
        "bearing_sin": dy / distance if distance > 0 else None,
        "orientation_difference_deg": orientation_difference,
        "height_difference_m": height_difference,
        "height_ratio": height_ratio,
        "footprint_area_ratio": area_ratio,
        "missing": {
            "bearing": distance <= 0,
            "orientation_difference_deg": orientation_difference is None,
            "height_difference_m": height_difference is None,
            "height_ratio": height_ratio is None,
            "footprint_area_ratio": area_ratio is None,
        },
    }


def _components(node_ids: list[str], pairs: set[tuple[str, str]]) -> int:
    if not node_ids:
        return 0
    adjacency: dict[str, set[str]] = {node_id: set() for node_id in node_ids}
    for first, second in pairs:
        adjacency[first].add(second)
        adjacency[second].add(first)
    unseen = set(node_ids)
    count = 0
    while unseen:
        count += 1
        start = unseen.pop()
        queue = deque([start])
        while queue:
            current = queue.popleft()
            for neighbour in adjacency[current]:
                if neighbour in unseen:
                    unseen.remove(neighbour)
                    queue.append(neighbour)
    return count


def build_edges(
    nodes: list[dict[str, Any]], *, scale_m: float
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build one undirected edge set without perturbing degenerate coordinates."""
    if not math.isfinite(scale_m) or scale_m <= 0:
        raise ValueError("scale_m must be finite and greater than zero")
    count = len(nodes)
    base = {
        "node_count": count,
        "duplicate_point_groups": [],
        "connected_components": count if count else 0,
        "average_degree": 0.0,
    }
    if count <= 1:
        return [], {**base, "edge_construction_status": "single_node" if count else "empty"}

    coordinates = np.asarray(
        [
            [node["representative_point_x_m"], node["representative_point_y_m"]]
            for node in nodes
        ],
        dtype=float,
    )
    point_groups: dict[tuple[float, float], list[str]] = defaultdict(list)
    for item, coordinate in zip(nodes, coordinates):
        point_groups[(float(coordinate[0]), float(coordinate[1]))].append(
            item["building_id"]
        )
    duplicates = [sorted(ids) for ids in point_groups.values() if len(ids) > 1]
    if duplicates:
        return [], {
            **base,
            "edge_construction_status": "degenerate_duplicate_points",
            "duplicate_point_groups": sorted(duplicates),
        }

    if count == 2:
        pairs = {(0, 1)}
        status = "two_node_rule"
    else:
        centered = coordinates - coordinates.mean(axis=0)
        if np.linalg.matrix_rank(centered) < 2:
            return [], {**base, "edge_construction_status": "degenerate_collinear"}
        try:
            triangulation = Delaunay(coordinates)
        except QhullError:
            return [], {**base, "edge_construction_status": "degenerate_delaunay_failure"}
        pairs: set[tuple[int, int]] = set()
        for simplex in triangulation.simplices:
            for first, second in (
                (simplex[0], simplex[1]),
                (simplex[1], simplex[2]),
                (simplex[2], simplex[0]),
            ):
                pairs.add(tuple(sorted((int(first), int(second)))))
        status = "delaunay"

    edges = [_edge_record(nodes[first], nodes[second], scale_m) for first, second in pairs]
    edges.sort(key=lambda item: (item["source"], item["target"]))
    id_pairs = {(edge["source"], edge["target"]) for edge in edges}
    components = _components([node["building_id"] for node in nodes], id_pairs)
    return edges, {
        **base,
        "edge_construction_status": status,
        "connected_components": components,
        "average_degree": 2.0 * len(edges) / count,
    }


def write_graph_json(graph: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(graph, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def read_graph_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _nonfinite_paths(value: Any, prefix: str = "") -> list[str]:
    paths: list[str] = []
    if isinstance(value, float) and not math.isfinite(value):
        paths.append(prefix)
    elif isinstance(value, dict):
        for key, item in value.items():
            paths.extend(_nonfinite_paths(item, f"{prefix}.{key}" if prefix else key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            paths.extend(_nonfinite_paths(item, f"{prefix}[{index}]"))
    return paths


def validate_graph(
    graph: dict[str, Any], *, expected_node_count: int
) -> dict[str, Any]:
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    ids = [node.get("building_id") for node in nodes]
    self_loops = sum(edge.get("source") == edge.get("target") for edge in edges)
    seen: set[tuple[str, str]] = set()
    duplicate_edges = 0
    negative_distances = 0
    for edge in edges:
        pair = tuple(sorted((edge.get("source"), edge.get("target"))))
        if pair in seen and pair[0] != pair[1]:
            duplicate_edges += 1
        seen.add(pair)
        distance = edge.get("centroid_distance_m")
        if distance is not None and distance < 0:
            negative_distances += 1
    nonfinite = _nonfinite_paths(graph)
    result = {
        "expected_node_count": int(expected_node_count),
        "actual_node_count": len(nodes),
        "declared_node_count": graph.get("graph", {}).get("node_count"),
        "unique_building_id_count": len(set(ids)),
        "null_building_id_count": sum(item is None for item in ids),
        "self_loop_count": self_loops,
        "duplicate_undirected_edge_count": duplicate_edges,
        "negative_edge_distance_count": negative_distances,
        "nonfinite_value_count": len(nonfinite),
        "nonfinite_value_paths": nonfinite,
    }
    result["passed"] = all(
        (
            len(nodes) == expected_node_count,
            graph.get("graph", {}).get("node_count") == len(nodes),
            len(set(ids)) == len(nodes),
            result["null_building_id_count"] == 0,
            self_loops == 0,
            duplicate_edges == 0,
            negative_distances == 0,
            not nonfinite,
        )
    )
    return result
