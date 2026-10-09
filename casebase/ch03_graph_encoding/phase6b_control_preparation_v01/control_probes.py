"""Seven local geometries, the existing encoder, and bounded explanatory records."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
from shapely.geometry import box

CH03 = Path(__file__).resolve().parent.parent
PHASE4B = CH03 / "phase4b_full_graph_batches_v032"
PHASE6A = CH03 / "phase6a_relation_validation_setup_v01"
sys.path.insert(0, str(PHASE6A))
from model_inputs import source_value
from render_pairs import load_graph

PROBE_IDS = ("BASE", "GAP_080", "GAP_120", "OFFSET_08", "OFFSET_16", "HEIGHT_ADJ", "HEIGHT_SEP")
TARGET_FIELDS = {
    "gap": ("footprint_gap_m", "footprint_gap_block_norm", "centroid_distance_block_norm"),
    "offset": ("long_side_overlap_ratio", "long_side_offset_norm", "connection_axis_alignment_i", "connection_axis_alignment_j", "orientation_diff_deg"),
    "height": ("height_diff_m", "log_height_ratio_abs"),
}
FIELDS = tuple(dict.fromkeys(field for group in TARGET_FIELDS.values() for field in group))
GROUPS = {"gap": ("GAP_080", "BASE", "GAP_120"), "offset": ("BASE", "OFFSET_08", "OFFSET_16"),
          "height": ("HEIGHT_ADJ", "HEIGHT_SEP")}
EXAMPLE_PAIRS = {"gap": (("B01", "B02"), ("B01", "B05")),
                 "offset": (("B01", "B05"), ("B02", "B06")),
                 "height": (("B01", "B02"), ("B04", "B08"))}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_probe_layouts(spec: dict) -> list[dict]:
    definitions = spec["layouts"]
    if tuple(item["id"] for item in definitions) != PROBE_IDS or spec["srid"] != 0:
        raise ValueError("Use the seven specified local layouts and unspecified SRID 0")
    if len(spec["x_centers"]) != 4 or len(spec["y_centers"]) != 2:
        raise ValueError("The fixed probe requires two rows of four buildings")
    block = box(*spec["block_bounds"])
    width, depth = spec["building_dimensions"]
    ox, oy = spec["scale_origin"]
    definition_hash = hashlib.sha256(json.dumps(spec, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    layouts = []
    for definition in definitions:
        buildings = []
        for row, y in enumerate(spec["y_centers"]):
            for column, x in enumerate(spec["x_centers"]):
                building_id = f"B{row*4+column+1:02d}"
                cx, cy, height = float(x), float(y), float(spec["base_height_m"])
                operation = definition["operation"]
                if operation == "center_scale":
                    cx, cy = ox + (x-ox)*definition["factor"], oy + (y-oy)*definition["factor"]
                elif operation == "row_offset":
                    cx += definition["row_x_shifts"][row]
                elif operation == "height_assignment":
                    height = definition["high_m"] if building_id in definition["high_ids"] else definition["other_m"]
                elif operation != "base":
                    raise ValueError(f"Unknown operation: {operation}")
                geometry = box(cx-width/2, cy-depth/2, cx+width/2, cy+depth/2)
                if not block.contains(geometry) or not math.isfinite(height) or height <= 0:
                    raise ValueError(f"Probe building outside block or invalid height: {building_id}")
                buildings.append({"building_id": building_id, "citygml_id": f"synthetic_local:{building_id}",
                                  "geometry": geometry, "area": float(geometry.area),
                                  "perimeter": float(geometry.length), "height": float(height)})
        if any(a["geometry"].intersects(b["geometry"]) for i,a in enumerate(buildings) for b in buildings[i+1:]):
            raise ValueError("Probe footprints intersect; do not repair or search for new parameters")
        layouts.append({"id": definition["id"], "block_geometry": block, "buildings": buildings,
                        "definition": definition, "definition_sha256": definition_hash,
                        "tessellation": dict(spec["tessellation"]), "crs_description": spec["crs_description"]})
    return layouts


def encode_probe(layout: dict) -> dict:
    # Import the encoder only, not run_phase4b or a database runner. Old modules have
    # database-related imports, but this pure geometry call never requests a connection.
    sys.path.insert(0, str(PHASE4B))
    from export_phase4b_v032 import encode_block_to_v032, TessellationConfig
    block_geometry = layout["block_geometry"]
    buildings = layout["buildings"]
    area_total = sum(b["area"] for b in buildings)
    sample = {"sample_id": layout["id"], "block_id": layout["id"], "city": "synthetic_local",
              "source_kind": "synthetic_local"}
    block = {"geometry": block_geometry, "block_id": layout["id"], "city": "synthetic_local",
             "city_query": "synthetic_local", "country": None, "storage_srid": 0, "metric_srid": 0,
             "area_m2": float(block_geometry.area), "lod1_building_count": len(buildings),
             "bcr": area_total/block_geometry.area,
             "avh_w": sum(b["area"]*b["height"] for b in buildings)/area_total}
    outcome = encode_block_to_v032(sample, block, buildings, sample_manifest_sha256=layout["definition_sha256"],
                                   tessellation_config=TessellationConfig(**layout["tessellation"]))
    if outcome.status != "completed" or outcome.record is None or outcome.excluded_buildings:
        raise ValueError(f"Unexpected building exclusion in local probe {layout['id']}: {outcome.excluded_buildings}")
    record = outcome.record
    if not record["validation"]["passed"]:
        raise ValueError(f"Probe validation failed: {record['validation']['issues']}")
    # Correct only provenance generated by the old database-oriented adapter. No
    # geometry, node feature, edge, flag or validation result is edited here.
    record["experiment_id"] = "CH03-P6B-PREP-CONTROL-V01"
    record["source_kind"] = "synthetic_local"
    record["probe_definition"] = {"operation": layout["definition"], "units": "m", "srid": 0,
                                  "crs_description": layout["crs_description"], "not_a_generated_model_result": True}
    record["graph"]["source_index"] = {"source_kind": "synthetic_local", "definition_file": "probe_layouts_v01.json",
                                      "block_id": layout["id"], "geometry_source": "local_rectangles"}
    record["graph"]["crs_description"] = layout["crs_description"]
    for node in record["nodes"]:
        node["source_index"] = {"source_kind": "synthetic_local", "schema": None, "relation": "probe_layouts_v01.json",
                                "geometry_column": "generated_rectangle", "building_id": node["building_id"]}
    attributes = record["block_attributes"]
    attributes["missing_reasons"] = {field: "not_calculated_for_synthetic_probe" if missing else None
                                     for field, missing in attributes["missing"].items()}
    return record


def summarize_probe(record: dict) -> dict:
    attributes = record.get("block_attributes", {})
    values = {**attributes.get("values", {}), "missing": attributes.get("missing", {})}
    nodes = record.get("nodes", [])
    areas = [source_value(node, "footprint_area_m2") for node in nodes]
    heights = [source_value(node, "height_m") for node in nodes]
    complete = bool(nodes) and all(a is not None and h is not None for a,h in zip(areas, heights, strict=True))
    total = sum(areas) if nodes and all(a is not None for a in areas) else None
    mean = sum(a*h for a,h in zip(areas, heights, strict=True))/total if complete and total > 0 else None
    result = {"layout_id": record.get("sample", {}).get("sample_id"), "status": "completed",
              "node_count": len(nodes), "edge_count": len(record.get("edges", [])),
              "area_m2": source_value(values, "area_m2"), "footprint_area_total_m2": total,
              "avh_w_m": mean, "bcr": source_value(values, "bcr"),
              "cell_failure_count": len(record.get("cell_failures", [])), "error": ""}
    for field in FIELDS:
        available = [v for edge in record.get("edges", []) if (v := source_value(edge, field)) is not None]
        q1, median, q3 = np.quantile(available, (0.25, 0.5, 0.75)) if available else (None, None, None)
        result[f"{field}_valid_count"] = len(available)
        result[f"{field}_median"] = float(median) if median is not None else None
        result[f"{field}_iqr"] = float(q3-q1) if q1 is not None else None
    return result


def edge_comparisons(records: dict, target: str) -> list[dict]:
    rows = []
    for a,b in EXAMPLE_PAIRS[target]:
        matches = {name: next((e for e in record.get("edges", []) if {str(e["source"]), str(e["target"])} == {a,b}), None)
                   for name,record in records.items()}
        changed = any(e is None for e in matches.values()) and any(e is not None for e in matches.values())
        for name, edge in matches.items():
            rows.append({"target": target, "layout_id": name, "endpoint_a": a, "endpoint_b": b,
                         "present": edge is not None, "adjacency_note": "邻接改变" if changed else ("已有邻接" if edge else "该对均非邻接"),
                         "source": edge.get("source") if edge else None, "target_endpoint": edge.get("target") if edge else None,
                         **{field: source_value(edge, field) if edge else None for field in FIELDS},
                         "missing_reasons": json.dumps(edge.get("missing_reasons", {}) if edge else {"edge": "not_adjacent"}, ensure_ascii=False)})
    return rows


def select_real_cases(manifest_path: Path, graph_root: Path) -> tuple[dict, list[dict]]:
    """At most 60 known files, no graph-directory walk and no geometric matching."""
    with manifest_path.open(encoding="utf-8-sig", newline="") as handle:
        pairs = list(csv.DictReader(handle))
    candidates = {}
    for pair in pairs:
        for side in ("a", "b"):
            entry = {"city": pair["city"], **{field: pair[f"{field}_{side}"] for field in ("sample_id", "block_id", "graph_path")}}
            sample_id = entry["sample_id"]
            if sample_id in candidates and candidates[sample_id] != entry:
                raise ValueError("Manifest has inconsistent duplicate sample identities")
            candidates[sample_id] = entry
    if len(candidates) > 60:
        raise ValueError("Phase 6B preparation only accepts the already-viewed 60-case pool")
    cards = {target: {"record": None, "reason": "已展示案例中未找到两条可读关系记录；不扩展抽样", "examples": [], "source": None}
             for target in TARGET_FIELDS}
    sources = []
    for entry in candidates.values():
        record = load_graph(entry, graph_root)  # Reuses Phase 4B-relative escape/identity checks.
        path = (graph_root / entry["graph_path"]).resolve()
        sources.append({**entry, "sha256": file_sha256(path)})
        for target, fields in TARGET_FIELDS.items():
            if cards[target]["record"] is not None:
                continue
            edges = [e for e in sorted(record.get("edges", []), key=lambda e: str(e["edge_id"]))
                     if all(source_value(e, field) is not None for field in fields)]
            if target == "height":
                edges = [e for e in edges if source_value(e, "height_diff_m") > 0]
            if len(edges) >= 2:
                cards[target] = {"record": record, "examples": edges[:2], "source": entry,
                                 "reason": "按当前清单顺序取首个有两条有效记录的案例，仅供阅读，未自动确认组织类型"}
        if all(card["record"] is not None for card in cards.values()):
            break
    return cards, sources
