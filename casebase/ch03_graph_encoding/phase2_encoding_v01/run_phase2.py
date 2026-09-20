"""Build the Chapter 3 phase 2 sample graphs from read-only database queries."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon as MplPolygon
from psycopg2 import sql
from shapely import from_wkb, wkt
from shapely.geometry import MultiPolygon, Polygon

from graph_encoding import (
    ENCODING_VERSION,
    build_edges,
    derive_node,
    read_graph_json,
    validate_graph,
    write_graph_json,
)


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


SAMPLE_DEFINITIONS = [
    {
        "sample_id": "trace_berlin_small",
        "sample_category": "常规追溯",
        "sample_subtype": "Berlin约10%建筑数分位",
        "city": "Berlin",
        "block_id": "DE_BE_002071",
        "selection_reason": "阶段1已核实的Berlin约10%建筑数分位样本",
        "is_boundary": False,
    },
    {
        "sample_id": "trace_berlin_medium",
        "sample_category": "常规追溯",
        "sample_subtype": "Berlin约50%建筑数分位",
        "city": "Berlin",
        "block_id": "DE_BE_010880",
        "selection_reason": "阶段1已核实的Berlin约50%建筑数分位样本",
        "is_boundary": False,
    },
    {
        "sample_id": "trace_berlin_large",
        "sample_category": "常规追溯",
        "sample_subtype": "Berlin约90%建筑数分位",
        "city": "Berlin",
        "block_id": "DE_BE_005279",
        "selection_reason": "阶段1已核实的Berlin约90%建筑数分位样本",
        "is_boundary": False,
    },
    {
        "sample_id": "trace_luxembourg_small",
        "sample_category": "常规追溯",
        "sample_subtype": "Luxembourg约10%建筑数分位",
        "city": "Luxembourg",
        "block_id": "LU_LU_000408",
        "selection_reason": "阶段1已核实的Luxembourg约10%建筑数分位样本",
        "is_boundary": False,
    },
    {
        "sample_id": "trace_luxembourg_medium",
        "sample_category": "常规追溯",
        "sample_subtype": "Luxembourg约50%建筑数分位",
        "city": "Luxembourg",
        "block_id": "LU_LU_000326",
        "selection_reason": "阶段1已核实的Luxembourg约50%建筑数分位样本",
        "is_boundary": False,
    },
    {
        "sample_id": "trace_luxembourg_large",
        "sample_category": "常规追溯",
        "sample_subtype": "Luxembourg约90%建筑数分位",
        "city": "Luxembourg",
        "block_id": "LU_LU_000454",
        "selection_reason": "阶段1已核实的Luxembourg约90%建筑数分位样本",
        "is_boundary": False,
    },
    {
        "sample_id": "trace_tokyo_small",
        "sample_category": "常规追溯",
        "sample_subtype": "Tokyo约10%建筑数分位",
        "city": "Tokyo",
        "block_id": "JP_TK_047746",
        "selection_reason": "阶段1已核实的Tokyo约10%建筑数分位样本",
        "is_boundary": False,
    },
    {
        "sample_id": "trace_tokyo_medium",
        "sample_category": "常规追溯",
        "sample_subtype": "Tokyo约50%建筑数分位",
        "city": "Tokyo",
        "block_id": "JP_TK_054219",
        "selection_reason": "阶段1已核实的Tokyo约50%建筑数分位样本",
        "is_boundary": False,
    },
    {
        "sample_id": "trace_tokyo_large",
        "sample_category": "常规追溯",
        "sample_subtype": "Tokyo约90%建筑数分位",
        "city": "Tokyo",
        "block_id": "JP_TK_058267",
        "selection_reason": "阶段1已核实的Tokyo约90%建筑数分位样本",
        "is_boundary": False,
    },
    {
        "sample_id": "contrast_high_coverage",
        "sample_category": "形态对照",
        "sample_subtype": "高覆盖且紧凑",
        "city": "Osaka",
        "block_id": "JP_OS_004938",
        "selection_reason": "排除阶段1三城后，在10至120栋、BCR 0.55至1且compactness≥0.45的候选中，按BCR和compactness降序首位",
        "is_boundary": False,
    },
    {
        "sample_id": "contrast_low_coverage",
        "sample_category": "形态对照",
        "sample_subtype": "低覆盖且疏散",
        "city": "Winnipeg",
        "block_id": "CA_WP_000972",
        "selection_reason": "排除阶段1三城后，在10至120栋且0<BCR≤0.05的候选中，平均最近邻距离最大",
        "is_boundary": False,
    },
    {
        "sample_id": "contrast_orientation_irregular",
        "sample_category": "形态对照",
        "sample_subtype": "朝向分散且边界较不规则",
        "city": "Portland",
        "block_id": "US_PO_001620",
        "selection_reason": "排除阶段1三城并保持与另两类城市不同后，在10至120栋候选中选择高orientation_entropy与较低rectangularity组合",
        "is_boundary": False,
    },
    {
        "sample_id": "boundary_single",
        "sample_category": "边界情况",
        "sample_subtype": "单建筑街区",
        "city": "Amsterdam",
        "block_id": "NL_AM_000241",
        "selection_reason": "建筑数为1、街区几何有效、面积为正且BCR处于0.1至0.8的确定性首个候选",
        "is_boundary": True,
    },
    {
        "sample_id": "boundary_large",
        "sample_category": "边界情况",
        "sample_subtype": "不少于500建筑对象",
        "city": "Sapporo",
        "block_id": "JP_SP_026935",
        "selection_reason": "建筑数不少于500且街区几何有效的最小建筑数候选，恰为500栋",
        "is_boundary": True,
    },
    {
        "sample_id": "boundary_duplicate_centroid",
        "sample_category": "边界情况",
        "sample_subtype": "最近邻距离为0/重复质心信号",
        "city": "Berlin",
        "block_id": "DE_BE_011422",
        "selection_reason": "全候选表仅两个avg_nn_distance=0街区之一，取建筑数相同情况下block_id排序首位",
        "is_boundary": True,
    },
]


VISUAL_REVIEW = {
    "trace_berlin_small": ("通过", "局部组团及其间联系可辨认", "少量跨组边未妨碍基本组织判断"),
    "trace_berlin_medium": ("有条件通过", "行列和分段组织可辨认", "中央空隙存在数条偏长连接"),
    "trace_berlin_large": ("有条件通过", "外围组团和内部空地可辨认", "跨中央空地的长边较明显"),
    "trace_luxembourg_small": ("通过", "双节点关系与原始布局一致", "按双节点专门规则形成一条边"),
    "trace_luxembourg_medium": ("通过", "围绕开敞空间的组织仍可辨认", "未见反复的系统性错连"),
    "trace_luxembourg_large": ("通过", "组团、疏密和边界方向可辨认", "少量长边仍属稀疏区域的可解释连接"),
    "trace_tokyo_small": ("通过", "线性排列关系清楚", "边集保持局部且规模适中"),
    "trace_tokyo_medium": ("有条件通过", "狭长街区方向可辨认", "存在较长连接且有建筑底面跨出街区边界"),
    "trace_tokyo_large": ("有条件通过", "主体密集区与离散组团可辨认", "离散组团之间出现明显长边"),
    "contrast_high_coverage": ("有条件通过", "高覆盖、行列和围合关系可辨认", "部分建筑底面跨出街区边界，需保留源数据质量标记"),
    "contrast_low_coverage": ("不通过", "节点可表达极低覆盖与离散分布", "Delaunay多次跨越大面积空地，关系边不能稳定代表可理解近邻"),
    "contrast_orientation_irregular": ("不通过", "分叉边界与沿边建筑带可辨认", "Delaunay跨越凹形空地和分叉两端，形成反复的无意义长边"),
    "boundary_single": ("通过", "单节点位置与底面一致", "按规则保留为合法无边图"),
    "boundary_large": ("有条件通过", "500节点仍保持稀疏图而非全连接", "大空地周边仍出现较长边，图面可读性下降"),
    "boundary_duplicate_centroid": ("有条件通过", "重合对象被忠实保留", "重复质心导致Delaunay退化；程序未抖动、合并或删点"),
}


def freeze_phase1_files(phase1_dir: Path) -> list[dict[str, str]]:
    """Return a deterministic SHA256 inventory without modifying phase 1."""
    rows = []
    for path in sorted(item for item in phase1_dir.rglob("*") if item.is_file()):
        rows.append(
            {
                "path": path.relative_to(phase1_dir).as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    return rows


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


def fetch_blocks(connection):
    block_ids = [item["block_id"] for item in SAMPLE_DEFINITIONS]
    selected_columns = sql.SQL(", ").join(
        sql.SQL("v.{} AS {}").format(sql.Identifier(name), sql.Identifier(name))
        for name in METRIC_FIELDS
    )
    statement = sql.SQL(
        """
        SELECT v.block_id, v.city, v.country, m.city_query,
               m.lod1_metric_srid AS metric_srid,
               ST_SRID(v.geom) AS storage_srid,
               {selected_columns},
               ST_AsBinary(
                   ST_Transform(ST_Force2D(v.geom), m.lod1_metric_srid)
               ) AS geometry_wkb_metric
        FROM summary.lod1_candidate_blocks v
        JOIN summary.city_name_map m ON m.city = v.city
        WHERE v.block_id = ANY(%s)
        ORDER BY v.block_id;
        """
    ).format(selected_columns=selected_columns)
    with connection.cursor() as cursor:
        cursor.execute(statement, (block_ids,))
        columns = [item.name for item in cursor.description]
        rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
    blocks = {}
    for row in rows:
        geometry_wkb = row.pop("geometry_wkb_metric")
        row["geometry"] = from_wkb(bytes(geometry_wkb))
        blocks[row["block_id"]] = row
    missing = sorted(set(block_ids) - set(blocks))
    if missing:
        raise RuntimeError(f"Sample blocks not found: {', '.join(missing)}")
    return blocks


def fetch_buildings(connection, block):
    relation = f"{block['city_query']}_buildings_lod1"
    statement = sql.SQL(
        """
        SELECT building_id, citygml_id, height, area, perimeter,
               ST_AsBinary(
                   ST_Transform(ST_Force2D(geom_2d), %s)
               ) AS geometry_wkb_metric
        FROM lod1.{}
        WHERE block_id = %s
        ORDER BY building_id;
        """
    ).format(sql.Identifier(relation))
    with connection.cursor() as cursor:
        cursor.execute(statement, (block["metric_srid"], block["block_id"]))
        columns = [item.name for item in cursor.description]
        rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
    for row in rows:
        geometry_wkb = row.pop("geometry_wkb_metric")
        row["geometry"] = from_wkb(bytes(geometry_wkb))
    return rows


def _write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _format_number(value, digits=3):
    if value is None:
        return "NA"
    return f"{float(value):.{digits}f}"


def _polygon_parts(geometry):
    if isinstance(geometry, Polygon):
        return [geometry]
    if isinstance(geometry, MultiPolygon):
        return list(geometry.geoms)
    return [part for part in getattr(geometry, "geoms", []) if isinstance(part, Polygon)]


def _plot_outline(axis, geometry, **kwargs):
    for polygon in _polygon_parts(geometry):
        x, y = polygon.exterior.xy
        axis.plot(x, y, **kwargs)
        for ring in polygon.interiors:
            x, y = ring.xy
            axis.plot(x, y, **kwargs)


def _plot_footprints(axis, nodes, values=None, cmap="viridis"):
    patches = []
    colors = []
    for index, node in enumerate(nodes):
        geometry = wkt.loads(node["geometry_wkt_metric"])
        for polygon in _polygon_parts(geometry):
            patches.append(MplPolygon(np.asarray(polygon.exterior.coords), closed=True))
            if values is not None:
                colors.append(values[index])
    if not patches:
        return None
    collection = PatchCollection(
        patches,
        edgecolor="#4b5563",
        linewidth=0.35,
        facecolor="#d8dee9" if values is None else None,
        cmap=cmap if values is not None else None,
        alpha=0.9,
    )
    if values is not None:
        numeric = np.asarray(colors, dtype=float)
        masked = np.ma.masked_invalid(numeric)
        collection.set_array(masked)
        collection.set_clim(float(masked.min()), float(masked.max()))
    axis.add_collection(collection)
    axis.autoscale_view()
    return collection


def _plot_edges(axis, graph, normalized=False):
    node_by_id = {node["building_id"]: node for node in graph["nodes"]}
    for edge in graph["edges"]:
        first = node_by_id[edge["source"]]
        second = node_by_id[edge["target"]]
        if normalized:
            x = [first["local_x_normalized"], second["local_x_normalized"]]
            y = [first["local_y_normalized"], second["local_y_normalized"]]
        else:
            x = [first["representative_point_x_m"], second["representative_point_x_m"]]
            y = [first["representative_point_y_m"], second["representative_point_y_m"]]
        axis.plot(x, y, color="#dc2626", linewidth=0.55, alpha=0.68, zorder=3)
    if normalized:
        x = [node["local_x_normalized"] for node in graph["nodes"]]
        y = [node["local_y_normalized"] for node in graph["nodes"]]
    else:
        x = [node["representative_point_x_m"] for node in graph["nodes"]]
        y = [node["representative_point_y_m"] for node in graph["nodes"]]
    axis.scatter(x, y, s=7, color="#111827", zorder=4)


def plot_four_panel(graph, path, *, dpi=180):
    block = wkt.loads(graph["graph"]["block_geometry_wkt_metric"])
    figure, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
    panels = axes.ravel()

    _plot_footprints(panels[0], graph["nodes"])
    _plot_outline(panels[0], block, color="#111827", linewidth=1.2)
    panels[0].set_title("Footprints and block boundary")

    _plot_footprints(panels[1], graph["nodes"])
    _plot_outline(panels[1], block, color="#111827", linewidth=1.0)
    _plot_edges(panels[1], graph)
    panels[1].set_title("Footprints with relation edges")

    heights = [node["height_m"] for node in graph["nodes"]]
    use_height = any(value is not None for value in heights)
    values = [float(value) if value is not None else np.nan for value in heights]
    label = "Building height (m)"
    if not use_height:
        values = [node["footprint_area_m2"] for node in graph["nodes"]]
        label = "Footprint area (m2)"
    collection = _plot_footprints(panels[2], graph["nodes"], values=values, cmap="viridis")
    _plot_outline(panels[2], block, color="#111827", linewidth=1.0)
    panels[2].set_title(label)
    if collection is not None:
        figure.colorbar(collection, ax=panels[2], shrink=0.75)

    _plot_edges(panels[3], graph, normalized=True)
    panels[3].axhline(0, color="#9ca3af", linewidth=0.4)
    panels[3].axvline(0, color="#9ca3af", linewidth=0.4)
    panels[3].set_title("Translated and scale-normalized graph")

    for axis in panels:
        axis.set_aspect("equal", adjustable="datalim")
        axis.set_xticks([])
        axis.set_yticks([])
    values = graph["block_attributes"]["values"]
    flags = ", ".join(graph["graph"]["quality_flags"]) or "none"
    figure.suptitle(
        f"{graph['graph']['city']} | {graph['graph']['block_id']} | "
        f"n={graph['graph']['node_count']} | BCR={_format_number(values['bcr'])} | "
        f"FAR={_format_number(values['far'])}\nquality flags: {flags}",
        fontsize=12,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi, facecolor="white")
    plt.close(figure)


def plot_contact_sheet(graphs, path):
    figure, axes = plt.subplots(3, 5, figsize=(20, 12), constrained_layout=True)
    for axis, graph in zip(axes.ravel(), graphs):
        block = wkt.loads(graph["graph"]["block_geometry_wkt_metric"])
        _plot_footprints(axis, graph["nodes"])
        _plot_outline(axis, block, color="#111827", linewidth=0.8)
        _plot_edges(axis, graph)
        status = graph["diagnostics"]["edge_construction_status"]
        axis.set_title(
            f"{graph['graph']['city']}\n{graph['graph']['block_id']} | "
            f"n={graph['graph']['node_count']} | {status}",
            fontsize=8,
        )
        axis.set_aspect("equal", adjustable="datalim")
        axis.set_xticks([])
        axis.set_yticks([])
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.suptitle("Phase 2 sample graph contact sheet", fontsize=15)
    figure.savefig(path, dpi=180, facecolor="white")
    plt.close(figure)


def summarize_validation(graph, roundtrip_graph, expected_node_count):
    core = validate_graph(graph, expected_node_count=expected_node_count)
    edge_distances = [edge["centroid_distance_m"] for edge in graph["edges"]]
    median_edge = statistics.median(edge_distances) if edge_distances else None
    max_edge = max(edge_distances) if edge_distances else None
    ratio = max_edge / median_edge if median_edge and median_edge > 0 else None
    node_missing_consistent = all(
        (node.get("height_m") is None) == node["missing"]["height_m"]
        and (node.get("source_area_m2") is None)
        == node["missing"]["source_area_m2"]
        and (node.get("source_perimeter_m") is None)
        == node["missing"]["source_perimeter_m"]
        for node in graph["nodes"]
    )
    original_edges = [(edge["source"], edge["target"]) for edge in graph["edges"]]
    roundtrip_edges = [
        (edge["source"], edge["target"]) for edge in roundtrip_graph["edges"]
    ]
    roundtrip_consistent = (
        original_edges == roundtrip_edges
        and [node["building_id"] for node in graph["nodes"]]
        == [node["building_id"] for node in roundtrip_graph["nodes"]]
    )
    scale = graph["graph"]["coordinate_scale_m"]
    passed = bool(
        core["passed"]
        and node_missing_consistent
        and roundtrip_consistent
        and math.isfinite(scale)
        and scale > 0
    )
    return {
        **core,
        "edge_count": len(graph["edges"]),
        "edge_construction_status": graph["diagnostics"]["edge_construction_status"],
        "connected_components": graph["diagnostics"]["connected_components"],
        "average_degree": graph["diagnostics"]["average_degree"],
        "median_edge_length_m": median_edge,
        "longest_edge_length_m": max_edge,
        "longest_to_median_edge_ratio": ratio,
        "node_missing_flags_consistent": node_missing_consistent,
        "roundtrip_nodes_edges_consistent": roundtrip_consistent,
        "coordinate_scale_finite_positive": math.isfinite(scale) and scale > 0,
        "passed_core_checks": passed,
    }


def _git_value(repo_root, *arguments):
    completed = subprocess.run(
        ["git", "-c", f"safe.directory={repo_root.as_posix()}", *arguments],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def run():
    phase2_dir = Path(__file__).resolve().parent
    results_dir = phase2_dir / "results"
    ch03_dir = phase2_dir.parent
    repo_root = ch03_dir.parents[1]
    phase1_dir = ch03_dir / "phase1_audit"
    graphs_dir = results_dir / "graphs"
    visual_dir = results_dir / "visual_checks"
    sys.path.insert(0, str(ch03_dir))
    from shared.db_readonly import readonly_connection

    freeze_rows = freeze_phase1_files(phase1_dir)
    _write_csv(
        results_dir / "phase1_freeze_sha256.csv",
        freeze_rows,
        ["path", "sha256"],
    )

    started_at = datetime.now().astimezone().isoformat(timespec="seconds")
    with readonly_connection(statement_timeout_ms=300_000) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT current_database(), current_setting('transaction_read_only'),
                       clock_timestamp();
                """
            )
            database_name, transaction_read_only, database_snapshot_time = cursor.fetchone()
        blocks = fetch_blocks(connection)
        graphs = []
        for definition in SAMPLE_DEFINITIONS:
            block = blocks[definition["block_id"]]
            if block["city"] != definition["city"]:
                raise RuntimeError(
                    f"City mismatch for {block['block_id']}: "
                    f"{block['city']} != {definition['city']}"
                )
            buildings = fetch_buildings(connection, block)
            graphs.append(assemble_graph(definition, block, buildings))

    validation_rows = []
    manifest_rows = []
    visual_rows = []
    for graph in graphs:
        definition = graph["sample"]
        graph_path = graphs_dir / f"{definition['sample_id']}.graph.json"
        write_graph_json(graph, graph_path)
        reloaded = read_graph_json(graph_path)
        validation = summarize_validation(
            graph, reloaded, int(graph["block_attributes"]["values"]["lod1_building_count"])
        )
        is_degenerate = validation["edge_construction_status"].startswith("degenerate_")
        if validation["passed_core_checks"] and not is_degenerate:
            result = "通过"
        elif definition["is_boundary"] and validation["passed_core_checks"] and is_degenerate:
            result = "边界异常已明确保留"
        else:
            result = "不通过"
        validation_rows.append(
            {
                "sample_id": definition["sample_id"],
                "sample_category": definition["sample_category"],
                "city": graph["graph"]["city"],
                "block_id": graph["graph"]["block_id"],
                "is_boundary": definition["is_boundary"],
                "validation_result": result,
                **validation,
            }
        )
        values = graph["block_attributes"]["values"]
        manifest_rows.append(
            {
                **definition,
                "building_count": graph["graph"]["node_count"],
                "edge_count": graph["graph"]["edge_count"],
                "bcr": values["bcr"],
                "far": values["far"],
                "compactness": values["compactness"],
                "rectangularity": values["rectangularity"],
                "orientation_entropy": values["orientation_entropy"],
                "avg_nn_distance_m": values["avg_nn_distance"],
                "metric_srid": graph["graph"]["metric_srid"],
                "quality_flags": ";".join(graph["graph"]["quality_flags"]),
                "source_relation": graph["graph"]["source_index"]["building_relation"],
                "graph_file": graph_path.relative_to(phase2_dir).as_posix(),
                "encoding_result": result,
            }
        )
        ratio = validation["longest_to_median_edge_ratio"]
        judgement, spatial_expression, issue = VISUAL_REVIEW[definition["sample_id"]]
        visual_rows.append(
            {
                "sample_id": definition["sample_id"],
                "city": graph["graph"]["city"],
                "block_id": graph["graph"]["block_id"],
                "is_boundary": definition["is_boundary"],
                "judgement_result": judgement,
                "data_fidelity": "通过" if validation["passed_core_checks"] else "不通过",
                "spatial_expression": spatial_expression,
                "economy_stability": (
                    f"平均度={validation['average_degree']:.3f}; "
                    f"连通分量={validation['connected_components']}; "
                    f"最长/中位边长={_format_number(ratio)}"
                ),
                "judgement_basis": issue,
                "issue_location": validation["edge_construction_status"] if is_degenerate else issue,
                "visual_file": f"results/visual_checks/samples/{definition['sample_id']}.png",
                "researcher_review_required": True,
            }
        )
        plot_four_panel(
            graph,
            visual_dir / "samples" / f"{definition['sample_id']}.png",
            dpi=150,
        )

    representative_ids = {
        "contrast_high_coverage",
        "contrast_low_coverage",
        "contrast_orientation_irregular",
        "boundary_single",
        "boundary_large",
        "boundary_duplicate_centroid",
    }
    for graph in graphs:
        if graph["sample"]["sample_id"] in representative_ids:
            plot_four_panel(
                graph,
                visual_dir
                / "representative"
                / f"{graph['sample']['sample_id']}.png",
                dpi=220,
            )
    plot_contact_sheet(graphs, visual_dir / "contact_sheet_all_15.png")

    manifest_fields = list(SAMPLE_DEFINITIONS[0]) + [
        "building_count",
        "edge_count",
        "bcr",
        "far",
        "compactness",
        "rectangularity",
        "orientation_entropy",
        "avg_nn_distance_m",
        "metric_srid",
        "quality_flags",
        "source_relation",
        "graph_file",
        "encoding_result",
    ]
    _write_csv(results_dir / "sample_manifest.csv", manifest_rows, manifest_fields)
    validation_fields = list(validation_rows[0])
    _write_csv(
        results_dir / "validation_summary.csv", validation_rows, validation_fields
    )
    _write_csv(
        results_dir / "visual_judgement.csv",
        visual_rows,
        list(visual_rows[0]),
    )

    metadata = {
        "encoding_version": ENCODING_VERSION,
        "phase2_started_at": started_at,
        "phase2_finished_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "git_branch": _git_value(repo_root, "branch", "--show-current"),
        "git_head": _git_value(repo_root, "rev-parse", "HEAD"),
        "database_name": database_name,
        "database_snapshot_time": database_snapshot_time.isoformat(),
        "transaction_read_only": transaction_read_only,
        "sample_count": len(graphs),
        "regular_sample_count": sum(not item["is_boundary"] for item in SAMPLE_DEFINITIONS),
        "boundary_sample_count": sum(item["is_boundary"] for item in SAMPLE_DEFINITIONS),
        "phase1_frozen_file_count": len(freeze_rows),
        "database_writes": 0,
    }
    (results_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()
