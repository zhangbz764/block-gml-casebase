from __future__ import annotations

from copy import deepcopy


META = {
    "source_manifest_sha256": "source",
    "full_manifest_sha256": "full",
    "part_manifest_sha256": "part",
    "encoding_spec_sha256": "spec",
    "encoding_version": "0.3.2",
    "code_bundle_sha256": "code",
    "part_id": "part_01",
    "run_id": "full_v032",
    "parameter_sha256": "parameters",
}


def completed_status() -> dict:
    return {
        "sample_id": "S1", "city": "Test City", "country": "TC", "block_id": "B1",
        "part_id": "part_01", "status": "completed", "graph_path": "graphs/S1.graph.json.gz",
        "output_bytes": "123", "planned_building_count": "2", "excluded_building_count": "0",
    }


def normal_record() -> dict:
    nodes = [
        {"building_id": "A", "citygml_id": "A", "height_m": 10.0, "source_area_m2": 4.0,
         "source_perimeter_m": 8.0, "footprint_area_m2": 4.0, "footprint_perimeter_m": 8.0,
         "footprint_compactness": 0.8, "footprint_rectangularity": 1.0, "footprint_convexity": 1.0,
         "orientation_deg": 0.0, "orientation_sin_2theta": 0.0, "orientation_cos_2theta": 1.0,
         "representative_point_x_m": 1.0, "representative_point_y_m": 1.0, "local_x_m": 1.0,
         "local_y_m": 1.0, "local_x_normalized": -0.5, "local_y_normalized": 0.0,
         "boundary_distance_m": 1.0, "boundary_distance_normalized": 0.1,
         "geometry_wkt_metric": "POLYGON ((0 0,2 0,2 2,0 2,0 0))", "source_index": 0,
         "missing": {}, "missing_reasons": {}, "quality": {}},
        {"building_id": "B", "citygml_id": "B", "height_m": 12.0, "source_area_m2": 4.0,
         "source_perimeter_m": 8.0, "footprint_area_m2": 4.0, "footprint_perimeter_m": 8.0,
         "footprint_compactness": 0.8, "footprint_rectangularity": 1.0, "footprint_convexity": 1.0,
         "orientation_deg": 0.0, "orientation_sin_2theta": 0.0, "orientation_cos_2theta": 1.0,
         "representative_point_x_m": 4.0, "representative_point_y_m": 1.0, "local_x_m": 4.0,
         "local_y_m": 1.0, "local_x_normalized": 0.5, "local_y_normalized": 0.0,
         "boundary_distance_m": 1.0, "boundary_distance_normalized": 0.1,
         "geometry_wkt_metric": "POLYGON ((3 0,5 0,5 2,3 2,3 0))", "source_index": 1,
         "missing": {}, "missing_reasons": {}, "quality": {}},
    ]
    cells = [
        {"building_id": "A", "geometry_wkt_metric": "POLYGON ((0 0,3 0,3 3,0 3,0 0))", "area_m2": 9.0, "perimeter_m": 12.0, "geometry_valid": True},
        {"building_id": "B", "geometry_wkt_metric": "POLYGON ((3 0,6 0,6 3,3 3,3 0))", "area_m2": 9.0, "perimeter_m": 12.0, "geometry_valid": True},
    ]
    edge = {
        "edge_id": "A--B", "source": "A", "target": "B", "shared_cell_boundary_m": 3.0,
        "shared_boundary_ratio_min": 0.25, "shared_boundary_ratio_mean": 0.25,
        "footprint_gap_m": 1.0, "footprint_gap_block_norm": 0.1, "centroid_distance_m": 3.0,
        "centroid_distance_block_norm": 0.3, "orientation_diff_deg": 0.0, "orientation_similarity": 1.0,
        "long_side_overlap_ratio": 1.0, "long_side_offset_norm": 0.0,
        "connection_axis_alignment_i": 1.0, "connection_axis_alignment_j": 1.0,
        "log_area_ratio_abs": 0.0, "height_diff_m": 2.0, "log_height_ratio_abs": 0.18,
        "volume_ratio": 0.83, "missing": {}, "missing_reasons": {},
        "long_distance_candidate": False, "low_shared_candidate": False, "combined_weak_candidate": False,
    }
    values = {name: 1.0 for name in (
        "elongation compactness bcr convexity rectangularity far bcd osr avh_w hcv footprint_cv "
        "avg_complexity avg_rectangularity avg_convexity orientation_entropy area_m2 avh hsd h_max h_range "
        "total_volume avg_volume avg_footprint avg_orientation avg_nn_distance"
    ).split()}
    values["lod1_building_count"] = 2
    return {
        "encoding_version": "0.3.2", "sample": {"sample_id": "S1", "city": "Test City", "country": "TC", "block_id": "B1"},
        "phase4b": deepcopy(META),
        "graph": {"block_id": "B1", "city": "Test City", "country": "TC", "storage_srid": 4326,
                  "metric_srid": 3857, "block_geometry_wkt_metric": "POLYGON ((0 0,6 0,6 3,0 3,0 0))",
                  "source_index": {"block_relation": "blocks", "building_relation": "buildings"},
                  "source_building_count": 2, "encoded_building_count": 2, "excluded_building_count": 0,
                  "node_count": 2, "cell_count": 2, "cell_failure_count": 0, "edge_count": 1,
                  "quality_flags": []},
        "block_attributes": {"values": values, "missing": {}}, "nodes": nodes, "cells": cells,
        "cell_failures": [], "edges": [edge],
        "weak_edge_summary": {"long_distance_candidate_count": 0, "low_shared_candidate_count": 0,
                              "combined_weak_candidate_count": 0},
        "exclusions": {"buildings": [], "summary": {"source_building_count": 2,
                       "encoded_building_count": 2, "excluded_building_count": 0,
                       "excluded_building_reason_counts": {}}}, "validation": {"passed": True},
    }


def single_node_record() -> dict:
    record = normal_record()
    record["nodes"] = record["nodes"][:1]
    record["cells"] = record["cells"][:1]
    record["edges"] = []
    record["graph"].update({"source_building_count": 1, "encoded_building_count": 1,
                            "node_count": 1, "cell_count": 1, "edge_count": 0})
    record["exclusions"]["summary"].update({"source_building_count": 1, "encoded_building_count": 1})
    return record


def missing_and_exclusion_record() -> dict:
    record = single_node_record()
    record["graph"].update({"source_building_count": 2, "excluded_building_count": 1})
    record["exclusions"] = {"buildings": [{"building_id": "B", "reason": "invalid_geometry"}],
                            "summary": {"source_building_count": 2, "encoded_building_count": 1,
                                        "excluded_building_count": 1,
                                        "excluded_building_reason_counts": {"invalid_geometry": 1}}}
    record["nodes"][0]["height_m"] = None
    record["nodes"][0]["missing"] = {"height_m": True}
    record["nodes"][0]["missing_reasons"] = {"height_m": "source_null"}
    return record


def cell_failure_record() -> dict:
    record = normal_record()
    template_node = record["nodes"][0]
    template_cell = record["cells"][0]
    record["nodes"] = []
    record["cells"] = []
    for index in range(9):
        building_id = f"N{index}"
        node = deepcopy(template_node); node["building_id"] = building_id; node["citygml_id"] = building_id
        node["source_index"] = {"building_id": building_id}
        record["nodes"].append(node)
        if index < 8:
            cell = deepcopy(template_cell); cell["building_id"] = building_id
            record["cells"].append(cell)
    record["cell_failures"] = [{"building_id": "N8", "reason": "degenerate_cell"}]
    record["edges"] = []
    record["graph"].update({"source_building_count": 9, "encoded_building_count": 9,
                            "node_count": 9, "cell_count": 8, "cell_failure_count": 1, "edge_count": 0})
    record["exclusions"]["summary"].update({"source_building_count": 9, "encoded_building_count": 9})
    record["block_attributes"]["values"]["lod1_building_count"] = 9
    return record
