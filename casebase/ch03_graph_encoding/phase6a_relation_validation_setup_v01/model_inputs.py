"""Extract fair common inputs from a few existing Graphs; no geometry recomputation."""

from __future__ import annotations

from copy import deepcopy
import math


BLOCK_FIELDS = tuple("elongation compactness bcr convexity rectangularity far bcd osr avh_w hcv footprint_cv avg_complexity avg_rectangularity avg_convexity orientation_entropy".split())
NODE_CORE_FIELDS = tuple("footprint_compactness footprint_rectangularity footprint_convexity orientation_sin_2theta orientation_cos_2theta local_x_normalized local_y_normalized boundary_distance_normalized".split())
NODE_DERIVED_FIELDS = ("footprint_area_block_norm", "height_scale_norm", "approx_volume_block_norm")
EDGE_FIELDS = tuple("shared_boundary_ratio_min shared_boundary_ratio_mean footprint_gap_block_norm centroid_distance_block_norm orientation_similarity long_side_overlap_ratio long_side_offset_norm connection_axis_alignment_i connection_axis_alignment_j log_area_ratio_abs log_height_ratio_abs volume_ratio".split())


def finite_number(value) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (ValueError, TypeError):
        return None


def source_value(source: dict, field: str) -> float | None:
    if source.get("missing", {}).get(field) is True:
        return None
    return finite_number(source.get(field))


def divide(value: float | None, denominator: float | None) -> float | None:
    if value is None or denominator is None or denominator <= 0:
        return None
    return finite_number(value / denominator)


def extract_model_inputs(record: dict) -> dict:
    attributes = record.get("block_attributes", {})
    block_source = {**attributes.get("values", {}), "missing": attributes.get("missing", {})}
    block_values = {field: source_value(block_source, field) for field in BLOCK_FIELDS}
    block = {"fields": list(BLOCK_FIELDS), "values": block_values,
             "missing_mask": {field: value is None for field, value in block_values.items()}}
    nodes = sorted(record.get("nodes", []), key=lambda node: str(node["building_id"]))
    node_ids = [str(node["building_id"]) for node in nodes]
    if len(set(node_ids)) != len(node_ids):
        raise ValueError("Duplicate building IDs")
    area = source_value(block_source, "area_m2")
    scale = finite_number(record.get("graph", {}).get("coordinate_scale_m"))
    conditions_values = {"area_m2": area, "node_count": float(len(nodes)), "coordinate_scale_m": scale}
    conditions = {"values": conditions_values, "missing_mask": {key: value is None for key, value in conditions_values.items()}}
    values, masks, raw_values, reasons = [], [], [], []
    for node in nodes:
        footprint = source_value(node, "footprint_area_m2")
        height = source_value(node, "height_m")
        volume = finite_number(footprint * height) if footprint is not None and height is not None else None
        denominator = finite_number(area * scale) if area is not None and scale is not None else None
        row = {field: source_value(node, field) for field in NODE_CORE_FIELDS}
        row.update({"footprint_area_block_norm": divide(footprint, area), "height_scale_norm": divide(height, scale),
                    "approx_volume_block_norm": divide(volume, denominator)})
        values.append(row); masks.append({field: value is None for field, value in row.items()})
        raw_values.append({"footprint_area_m2": footprint, "height_m": height, "approx_volume_m3": volume})
        reasons.append({field: node.get("missing_reasons", {}).get(field, "source_missing_or_invalid_denominator")
                        for field, value in row.items() if value is None})
    shared_nodes = {"fields": [*NODE_CORE_FIELDS, *NODE_DERIVED_FIELDS], "node_ids": node_ids,
                    "values": values, "missing_mask": masks, "raw_values": raw_values, "missing_reasons": reasons}
    common = {"block": block, "conditions": conditions, "nodes": shared_nodes}
    graph = deepcopy(common)
    lookup = {building_id: i for i, building_id in enumerate(node_ids)}
    graph.update({"edge_fields": list(EDGE_FIELDS), "edge_index": [], "edge_values": [], "edge_missing_mask": [], "edge_ids": []})
    for edge in sorted(record.get("edges", []), key=lambda item: str(item["edge_id"])):
        source, target = str(edge["source"]), str(edge["target"])
        if source not in lookup or target not in lookup or source == target:
            raise ValueError("Edge endpoint does not map to two distinct encoded nodes")
        row = {field: source_value(edge, field) for field in EDGE_FIELDS}
        graph["edge_index"].append([lookup[source], lookup[target]])
        graph["edge_values"].append(row)
        graph["edge_missing_mask"].append({field: value is None for field, value in row.items()})
        graph["edge_ids"].append(str(edge["edge_id"]))
    return {
        "trace": {**record.get("sample", {}), "node_ids_are_mapping_only": True},
        "indicator": {"block": deepcopy(block), "conditions": deepcopy(conditions)},
        "building_set": deepcopy(common), "relation_graph": graph,
    }


def expand_bidirectional(graph: dict) -> dict:
    output = {"edge_index": [], "edge_values": [], "edge_missing_mask": []}
    for endpoints, row, mask in zip(graph["edge_index"], graph["edge_values"], graph["edge_missing_mask"], strict=True):
        reverse, reverse_mask = dict(row), dict(mask)
        i, j = "connection_axis_alignment_i", "connection_axis_alignment_j"
        reverse[i], reverse[j] = row[j], row[i]
        reverse_mask[i], reverse_mask[j] = mask[j], mask[i]
        output["edge_index"].extend([list(endpoints), list(reversed(endpoints))])
        output["edge_values"].extend([dict(row), reverse])
        output["edge_missing_mask"].extend([dict(mask), reverse_mask])
    return output
