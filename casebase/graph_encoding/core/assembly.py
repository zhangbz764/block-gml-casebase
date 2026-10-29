"""Pure graph assembly extracted without changing feature definitions."""
from __future__ import annotations
import math
from .building_features import ENCODING_VERSION,build_edges,derive_node

METRIC_FIELDS = [
    "area_m2",
    "elongation",
    "compactness",
    "lod1_building_count",
    "bcr",
    "convexity",
    "rectangularity",
    "far",
    "bcd",
    "osr",
    "avh",
    "avh_w",
    "hsd",
    "hcv",
    "h_max",
    "h_range",
    "total_volume",
    "avg_volume",
    "avg_footprint",
    "footprint_cv",
    "avg_complexity",
    "avg_rectangularity",
    "avg_convexity",
    "avg_orientation",
    "orientation_entropy",
    "avg_nn_distance",
]


def _number(value):
    if value is None:
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def assemble_graph(definition, block, buildings):
    """Assemble one model-neutral graph from projected Shapely geometries."""
    block_geometry = block["geometry"]
    area_m2 = _number(block.get("area_m2"))
    if area_m2 is None or area_m2 <= 0:
        raise ValueError(f"Block {block['block_id']} has no positive area_m2")
    scale_m = math.sqrt(area_m2)
    origin = block_geometry.centroid
    nodes = [
        derive_node(
            building,
            block_geometry=block_geometry,
            origin_x_m=float(origin.x),
            origin_y_m=float(origin.y),
            scale_m=scale_m,
        )
        for building in buildings
    ]
    edges, diagnostics = build_edges(nodes, scale_m=scale_m)
    for node in nodes:
        node["source_index"].update(
            {
                "schema": "lod1",
                "relation": f"{block['city_query']}_buildings_lod1",
            }
        )
    metric_values = {name: _number(block.get(name)) for name in METRIC_FIELDS}
    bcr = metric_values["bcr"]
    avg_nn = metric_values["avg_nn_distance"]
    quality_flags = []
    if bcr is not None and bcr > 1:
        quality_flags.append("bcr_gt_1")
    if avg_nn == 0:
        quality_flags.append("avg_nn_distance_zero")
    if diagnostics["edge_construction_status"].startswith("degenerate_"):
        quality_flags.append(diagnostics["edge_construction_status"])
    if any(not node["quality"]["centroid_covered_by_block"] for node in nodes):
        quality_flags.append("building_centroid_outside_block")
    if any(not node["quality"]["footprint_covered_by_block"] for node in nodes):
        quality_flags.append("building_footprint_crosses_block_boundary")

    return {
        "encoding_version": ENCODING_VERSION,
        "sample": dict(definition),
        "graph": {
            "block_id": block["block_id"],
            "city": block["city"],
            "country": block.get("country"),
            "lod": "LOD1",
            "storage_srid": int(block["storage_srid"]),
            "metric_srid": int(block["metric_srid"]),
            "node_count": len(nodes),
            "edge_count": len(edges),
            "local_origin_x_m": float(origin.x),
            "local_origin_y_m": float(origin.y),
            "coordinate_scale_m": scale_m,
            "quality_flags": quality_flags,
            "block_geometry_wkt_metric": block_geometry.wkt,
            "source_index": {
                "block_relation": "summary.lod1_candidate_blocks",
                "block_geometry_column": "geom",
                "block_id": block["block_id"],
                "building_relation": f"lod1.{block['city_query']}_buildings_lod1",
                "building_geometry_column": "geom_2d",
            },
        },
        "block_attributes": {
            "values": metric_values,
            "missing": {name: value is None for name, value in metric_values.items()},
        },
        "nodes": nodes,
        "edges": edges,
        "diagnostics": diagnostics,
    }
