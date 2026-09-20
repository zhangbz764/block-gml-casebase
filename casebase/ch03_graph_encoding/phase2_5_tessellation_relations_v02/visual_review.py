"""Generate phase 2.5 static, interactive, and sensitivity review artifacts."""

from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from shapely import wkt
from shapely.geometry import MultiPolygon, Polygon, mapping
from shapely.plotting import plot_polygon

from tessellation_encoding import (
    TessellationConfig,
    build_cell_adjacency,
    build_morphological_cells,
    read_json,
)


PHASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PHASE_DIR / "results"
SAMPLE_DIR = RESULTS_DIR / "visual_checks" / "samples"
REPRESENTATIVE_DIR = RESULTS_DIR / "visual_checks" / "representative"
SENSITIVITY_DIR = RESULTS_DIR / "visual_checks" / "sensitivity"
REPRESENTATIVE_SAMPLES = [
    "trace_berlin_medium",
    "boundary_large",
    "contrast_low_coverage",
    "contrast_orientation_irregular",
    "contrast_high_coverage",
]
SENSITIVITY_SAMPLES = REPRESENTATIVE_SAMPLES[:4]


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def polygon_parts(geometry):
    if isinstance(geometry, Polygon):
        yield geometry
    elif isinstance(geometry, MultiPolygon):
        yield from geometry.geoms
    else:
        for part in getattr(geometry, "geoms", []):
            yield from polygon_parts(part)


def node_point(node: dict[str, Any]) -> tuple[float, float]:
    x = node.get("representative_point_x_m")
    y = node.get("representative_point_y_m")
    if isinstance(x, (int, float)) and isinstance(y, (int, float)):
        if math.isfinite(float(x)) and math.isfinite(float(y)):
            return float(x), float(y)
    point = wkt.loads(node["geometry_wkt_metric"]).centroid
    return float(point.x), float(point.y)


def extent(block) -> tuple[float, float, float, float]:
    min_x, min_y, max_x, max_y = block.bounds
    pad = max(max_x - min_x, max_y - min_y) * 0.04 or 1.0
    return min_x - pad, max_x + pad, min_y - pad, max_y + pad


def style_axis(ax, bounds):
    ax.set_xlim(bounds[0], bounds[1])
    ax.set_ylim(bounds[2], bounds[3])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#c8ccd0")
        spine.set_linewidth(0.6)


def draw_outline(ax, geometry, color="#263238", linewidth=1.2):
    for polygon in polygon_parts(geometry):
        x, y = polygon.exterior.xy
        ax.plot(x, y, color=color, linewidth=linewidth, zorder=6)
        for ring in polygon.interiors:
            x, y = ring.xy
            ax.plot(x, y, color=color, linewidth=linewidth, zorder=6)


def draw_buildings(ax, nodes, face="#424b54", alpha=0.85, edge="#20262b"):
    for node in nodes:
        geometry = wkt.loads(node["geometry_wkt_metric"])
        for polygon in polygon_parts(geometry):
            plot_polygon(
                polygon,
                ax=ax,
                add_points=False,
                color=face,
                edgecolor=edge,
                linewidth=0.35,
                alpha=alpha,
                zorder=5,
            )


def draw_cells(ax, cells, alpha=0.34):
    palette = plt.get_cmap("tab20")
    for index, cell in enumerate(cells):
        geometry = wkt.loads(cell["geometry_wkt_metric"])
        for polygon in polygon_parts(geometry):
            plot_polygon(
                polygon,
                ax=ax,
                add_points=False,
                color=palette(index % 20),
                edgecolor="#66757f",
                linewidth=0.32,
                alpha=alpha,
                zorder=2,
            )


def edge_segments(edges, points):
    segments, kept = [], []
    for edge in edges:
        if edge["source"] in points and edge["target"] in points:
            segments.append([points[edge["source"]], points[edge["target"]]])
            kept.append(edge)
    return segments, kept


def draw_edges(ax, edges, points, color="#d95f02", linewidth=0.8, alpha=0.75):
    segments, _ = edge_segments(edges, points)
    if segments:
        ax.add_collection(
            LineCollection(
                segments,
                colors=color,
                linewidths=linewidth,
                alpha=alpha,
                zorder=4,
            )
        )


def draw_value_edges(
    ax,
    edges,
    points,
    field,
    cmap,
    linewidths=1.25,
    default_color="#9e9e9e",
):
    segments, kept = edge_segments(edges, points)
    values = [edge.get(field) for edge in kept]
    valid = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not segments:
        return None
    if not valid:
        ax.add_collection(LineCollection(segments, colors=default_color, linewidths=linewidths, zorder=4))
        return None
    minimum, maximum = min(valid), max(valid)
    if math.isclose(minimum, maximum):
        maximum = minimum + 1.0
    norm = Normalize(minimum, maximum)
    color_map = plt.get_cmap(cmap)
    colors = [color_map(norm(float(value))) if value is not None else default_color for value in values]
    widths = linewidths
    if field == "orientation_diff_deg":
        widths = [0.55 + 2.2 * float(edge.get("long_side_overlap_ratio") or 0.0) for edge in kept]
    ax.add_collection(LineCollection(segments, colors=colors, linewidths=widths, alpha=0.85, zorder=4))
    return plt.cm.ScalarMappable(norm=norm, cmap=color_map)


def graph_context(graph):
    block = wkt.loads(graph["graph"]["block_geometry_wkt_metric"])
    points = {node["building_id"]: node_point(node) for node in graph["nodes"]}
    return block, points, extent(block)


def plot_four_panel(graph, path: Path):
    block, points, bounds = graph_context(graph)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.4), constrained_layout=True)
    for ax in axes:
        style_axis(ax, bounds)
        draw_outline(ax, block)
    draw_buildings(axes[0], graph["nodes"])
    axes[0].set_title("1  Original footprints")
    draw_cells(axes[1], graph["cells"])
    draw_buildings(axes[1], graph["nodes"], alpha=0.78)
    axes[1].set_title("2  Morphological cells")
    draw_buildings(axes[2], graph["nodes"], face="#66757f", alpha=0.72)
    draw_edges(axes[2], graph["source_v01_edges"], points, color="#4c78a8")
    axes[2].set_title("3  v0.1 centroid Delaunay")
    draw_buildings(axes[3], graph["nodes"], face="#66757f", alpha=0.72)
    draw_edges(axes[3], graph["edges"], points, color="#e17c05")
    axes[3].set_title("4  v0.2 cell adjacency")
    values = graph.get("block_attributes", {}).get("values", {})
    title = (
        f"{graph['sample']['sample_id']} | {graph['graph']['city']} | {graph['graph']['block_id']} | "
        f"buildings {len(graph['nodes'])} | BCR {values.get('bcr', float('nan')):.3f} | "
        f"FAR {values.get('far', float('nan')):.3f} | cells {len(graph['cells'])} | "
        f"edges {len(graph['source_v01_edges'])}→{len(graph['edges'])} | "
        f"flags: {', '.join(graph.get('quality_flags', [])) or 'none'}"
    )
    fig.suptitle(title, fontsize=10)
    fig.savefig(path, dpi=170, facecolor="white")
    plt.close(fig)


def plot_six_panel(graph, path: Path):
    block, points, bounds = graph_context(graph)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), constrained_layout=True)
    axes = axes.ravel()
    for ax in axes:
        style_axis(ax, bounds)
        draw_outline(ax, block)
    draw_buildings(axes[0], graph["nodes"])
    axes[0].set_title("1  Original plan")
    draw_cells(axes[1], graph["cells"], alpha=0.45)
    draw_buildings(axes[1], graph["nodes"], alpha=0.72)
    axes[1].set_title("2  Cells and shared boundaries")
    draw_buildings(axes[2], graph["nodes"], face="#66757f", alpha=0.7)
    draw_edges(axes[2], graph["edges"], points)
    axes[2].set_title("3  Candidate adjacency")
    draw_buildings(axes[3], graph["nodes"], face="#87939a", alpha=0.66)
    mapper = draw_value_edges(axes[3], graph["edges"], points, "footprint_gap_block_norm", "viridis")
    axes[3].set_title("4  Footprint gap / block scale")
    if mapper:
        fig.colorbar(mapper, ax=axes[3], fraction=0.04, pad=0.02)
    draw_buildings(axes[4], graph["nodes"], face="#87939a", alpha=0.66)
    mapper = draw_value_edges(axes[4], graph["edges"], points, "orientation_diff_deg", "plasma")
    axes[4].set_title("5  Orientation; width = long-side overlap")
    if mapper:
        fig.colorbar(mapper, ax=axes[4], fraction=0.04, pad=0.02)
    draw_buildings(axes[5], graph["nodes"], face="#87939a", alpha=0.66)
    mapper = draw_value_edges(axes[5], graph["edges"], points, "log_area_ratio_abs", "cividis")
    missing_height = [edge for edge in graph["edges"] if edge["height_diff_m"] is None]
    draw_edges(axes[5], missing_height, points, color="#cc79a7", linewidth=1.8, alpha=0.95)
    missing_nodes = [node for node in graph["nodes"] if not isinstance(node.get("height_m"), (int, float)) or not (node.get("height_m") or 0) > 0]
    if missing_nodes:
        xy = np.asarray([points[node["building_id"]] for node in missing_nodes])
        axes[5].scatter(xy[:, 0], xy[:, 1], s=16, color="#cc79a7", zorder=7)
    axes[5].set_title("6  Area ratio; magenta = missing height")
    if mapper:
        fig.colorbar(mapper, ax=axes[5], fraction=0.04, pad=0.02)
    fig.suptitle(
        f"Relation diagnostics — {graph['sample']['sample_id']} ({graph['graph']['city']}, {graph['graph']['block_id']})",
        fontsize=13,
    )
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)


def html_payload(graphs):
    payload = {}
    for graph in graphs:
        sample_id = graph["sample"]["sample_id"]
        nodes = {node["building_id"]: node for node in graph["nodes"]}
        points = {building_id: node_point(node) for building_id, node in nodes.items()}
        gaps = [edge["shared_boundary_ratio_min"] for edge in graph["edges"]]
        low_shared = float(np.quantile(gaps, 0.1)) if gaps else None
        long_threshold = graph["diagnostics"]["long_distance_candidate"]["threshold"]
        payload[sample_id] = {
            "meta": {
                "city": graph["graph"]["city"],
                "block_id": graph["graph"]["block_id"],
                "scope_class": graph["scope_class"],
                "quality_flags": graph.get("quality_flags", []),
                "node_count": len(graph["nodes"]),
                "cell_count": len(graph["cells"]),
                "v01_edge_count": len(graph["source_v01_edges"]),
                "v02_edge_count": len(graph["edges"]),
                "long_threshold": long_threshold,
                "low_shared_threshold": low_shared,
            },
            "block": mapping(wkt.loads(graph["graph"]["block_geometry_wkt_metric"])),
            "buildings": [
                {
                    "id": node["building_id"],
                    "geometry": mapping(wkt.loads(node["geometry_wkt_metric"])),
                    "point": points[node["building_id"]],
                    "height_m": node.get("height_m"),
                }
                for node in graph["nodes"]
            ],
            "cells": [
                {
                    "id": cell["building_id"],
                    "geometry": mapping(wkt.loads(cell["geometry_wkt_metric"])),
                }
                for cell in graph["cells"]
            ],
            "v01": [
                {"source": edge["source"], "target": edge["target"], "line": [points[edge["source"]], points[edge["target"]]]}
                for edge in graph["source_v01_edges"]
                if edge["source"] in points and edge["target"] in points
            ],
            "v02": [
                dict(edge, line=[points[edge["source"]], points[edge["target"]]])
                for edge in graph["edges"]
            ],
        }
    return payload


HTML_TEMPLATE = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Phase 2.5 visual review</title>
<style>
:root{font-family:Inter,Segoe UI,Arial,sans-serif;color:#20262b;background:#f5f6f7}body{margin:0}header{padding:14px 18px;background:#263238;color:white;display:flex;gap:18px;align-items:center;flex-wrap:wrap}header h1{font-size:18px;margin:0}.controls{padding:10px 18px;background:white;border-bottom:1px solid #d7dce0;display:flex;gap:14px;flex-wrap:wrap}.meta{padding:10px 18px;font-size:13px}.viewer{margin:0 18px 18px;background:white;border:1px solid #d7dce0;border-radius:6px;overflow:hidden}svg{width:100%;height:72vh;display:block}.block{fill:none;stroke:#111;stroke-width:1.8;vector-effect:non-scaling-stroke}.building{fill:#455a64;stroke:#1f292e;stroke-width:.55;vector-effect:non-scaling-stroke}.cell{fill:#56b4e9;fill-opacity:.18;stroke:#72838c;stroke-width:.4;vector-effect:non-scaling-stroke}.v01{stroke:#4c78a8;stroke-width:.7;opacity:.55;vector-effect:non-scaling-stroke}.v02{stroke:#e17c05;stroke-width:1;opacity:.75;vector-effect:non-scaling-stroke}.id{fill:#111;text-anchor:middle;pointer-events:none}.highlight{stroke:#cc79a7!important;stroke-width:3!important;opacity:1!important}.quality .block{stroke:#cc79a7;stroke-width:3}.tooltip{position:fixed;display:none;background:rgba(30,35,39,.94);color:white;padding:8px 10px;border-radius:4px;font-size:12px;white-space:pre;pointer-events:none;z-index:10}.note{font-size:12px;color:#56636b;padding:0 18px 12px}
</style></head><body>
<header><h1>Graph encoding v0.2 — offline visual review</h1><label>Sample <select id="sample"></select></label></header>
<div class="controls">
<label><input type="checkbox" id="block" checked> block</label><label><input type="checkbox" id="buildings" checked> buildings</label><label><input type="checkbox" id="cells" checked> cells</label><label><input type="checkbox" id="v01"> v0.1 edges</label><label><input type="checkbox" id="v02" checked> v0.2 edges</label><label><input type="checkbox" id="ids"> building IDs</label>
<label><input type="checkbox" id="long"> highlight long distance</label><label><input type="checkbox" id="low"> highlight low shared ratio</label><label><input type="checkbox" id="missing"> highlight missing height</label><label><input type="checkbox" id="quality"> highlight quality/scope cases</label>
</div><div id="meta" class="meta"></div><div class="viewer"><svg id="svg"></svg></div><div class="note">Projected coordinates are preserved. Highlights are review aids, not automatic semantic decisions.</div><div id="tip" class="tooltip"></div>
<script>const DATA=__DATA__;
const ns='http://www.w3.org/2000/svg', sel=document.getElementById('sample'), svg=document.getElementById('svg'), meta=document.getElementById('meta'), tip=document.getElementById('tip');
Object.keys(DATA).forEach(k=>{const o=document.createElement('option');o.value=k;o.textContent=k;sel.appendChild(o)});
const ids=['block','buildings','cells','v01','v02','ids','long','low','missing','quality'];ids.forEach(id=>document.getElementById(id).addEventListener('change',render));sel.addEventListener('change',render);
function polygons(g){if(g.type==='Polygon')return [g.coordinates];if(g.type==='MultiPolygon')return g.coordinates;return []}
function path(g){return polygons(g).map(poly=>poly.map(r=>r.map((p,i)=>(i?'L':'M')+p[0]+' '+p[1]).join(' ')+' Z').join(' ')).join(' ')}
function E(name,attrs,parent=svg){const e=document.createElementNS(ns,name);for(const [k,v] of Object.entries(attrs))e.setAttribute(k,v);parent.appendChild(e);return e}
function showTip(ev,e){tip.textContent=`${e.edge_id}\n${e.source} → ${e.target}\nshared: ${fmt(e.shared_cell_boundary_m)} m\ngap: ${fmt(e.footprint_gap_m)} m\norientation: ${fmt(e.orientation_diff_deg)}°\nlong-side overlap: ${fmt(e.long_side_overlap_ratio)}\nlog area ratio: ${fmt(e.log_area_ratio_abs)}\nlog height ratio: ${fmt(e.log_height_ratio_abs)}`;tip.style.display='block';tip.style.left=(ev.clientX+12)+'px';tip.style.top=(ev.clientY+12)+'px'}
function fmt(v){return v===null||v===undefined?'missing':Number(v).toFixed(4)}
function render(){const d=DATA[sel.value],m=d.meta;svg.replaceChildren();let pts=[];polygons(d.block).forEach(p=>p.forEach(r=>r.forEach(x=>pts.push(x))));let xs=pts.map(p=>p[0]),ys=pts.map(p=>p[1]),minx=Math.min(...xs),maxx=Math.max(...xs),miny=Math.min(...ys),maxy=Math.max(...ys),pad=Math.max(maxx-minx,maxy-miny)*.04;svg.setAttribute('viewBox',`${minx-pad} ${-(maxy+pad)} ${maxx-minx+2*pad} ${maxy-miny+2*pad}`);let root=E('g',{transform:'scale(1,-1)',class:(document.getElementById('quality').checked&&m.scope_class!=='core_or_boundary_case')?'quality':''});
if(document.getElementById('cells').checked)d.cells.forEach(c=>E('path',{d:path(c.geometry),class:'cell'},root));
if(document.getElementById('block').checked)E('path',{d:path(d.block),class:'block'},root);
if(document.getElementById('v01').checked)d.v01.forEach(e=>E('line',{x1:e.line[0][0],y1:e.line[0][1],x2:e.line[1][0],y2:e.line[1][1],class:'v01'},root));
if(document.getElementById('v02').checked)d.v02.forEach(e=>{let hi=(document.getElementById('long').checked&&m.long_threshold!==null&&e.footprint_gap_block_norm>m.long_threshold)||(document.getElementById('low').checked&&m.low_shared_threshold!==null&&e.shared_boundary_ratio_min<=m.low_shared_threshold)||(document.getElementById('missing').checked&&e.height_diff_m===null);let line=E('line',{x1:e.line[0][0],y1:e.line[0][1],x2:e.line[1][0],y2:e.line[1][1],class:'v02'+(hi?' highlight':'')},root);line.addEventListener('mousemove',ev=>showTip(ev,e));line.addEventListener('mouseleave',()=>tip.style.display='none')});
if(document.getElementById('buildings').checked)d.buildings.forEach(b=>E('path',{d:path(b.geometry),class:'building'},root));
if(document.getElementById('ids').checked){let size=Math.max(maxx-minx,maxy-miny)/115;d.buildings.forEach(b=>{let t=E('text',{x:b.point[0],y:-b.point[1],class:'id','font-size':size,transform:'scale(1,-1)'},root);t.textContent=b.id})}
meta.textContent=`${sel.value} | ${m.city} | ${m.block_id} | ${m.scope_class} | buildings ${m.node_count} | cells ${m.cell_count} | edges ${m.v01_edge_count} → ${m.v02_edge_count} | flags: ${m.quality_flags.join(', ')||'none'}`}
sel.value=Object.keys(DATA)[0];render();</script></body></html>'''


def prepare_graphs():
    with (RESULTS_DIR / "input_manifest.csv").open(encoding="utf-8-sig", newline="") as handle:
        order = [row["sample_id"] for row in csv.DictReader(handle)]
    graphs = []
    for sample_id in order:
        graph = read_json(RESULTS_DIR / "graphs" / f"{sample_id}.graph.json")
        source = read_json(PHASE_DIR.parents[2] / graph["source_v01"]["path"])
        graph["source_v01_edges"] = source["edges"]
        graphs.append(graph)
    return graphs


def make_contact_sheet(paths):
    fig, axes = plt.subplots(5, 3, figsize=(22, 18), constrained_layout=True)
    for ax, path in zip(axes.ravel(), paths):
        ax.imshow(plt.imread(path))
        ax.set_title(path.stem.replace("_four_panel", ""), fontsize=9)
        ax.axis("off")
    fig.suptitle("Phase 2.5 — all 15 samples (four-panel review)", fontsize=16)
    fig.savefig(RESULTS_DIR / "visual_checks" / "contact_sheet.png", dpi=130, facecolor="white")
    plt.close(fig)


def sensitivity(graph_by_id):
    rows = []
    for sample_id in SENSITIVITY_SAMPLES:
        base_graph = graph_by_id[sample_id]
        block = wkt.loads(base_graph["graph"]["block_geometry_wkt_metric"])
        base_cells = {cell["building_id"]: cell for cell in base_graph["cells"]}
        base_edges = {edge["edge_id"]: edge for edge in base_graph["edges"]}
        variants = []
        for segment_m in (0.5, 1.0, 2.0):
            started = time.perf_counter()
            config = TessellationConfig(
                shrink_m=0.4,
                segment_m=segment_m,
                min_shared_boundary_m=1e-6,
            )
            cells, failures, diagnostics = build_morphological_cells(
                block, base_graph["nodes"], config
            )
            edges, _ = build_cell_adjacency(
                cells, base_graph["nodes"], block.area, config
            )
            point_count = diagnostics["sample_point_count"]
            runtime = time.perf_counter() - started
            cell_by_id = {cell["building_id"]: cell for cell in cells}
            edge_by_id = {edge["edge_id"]: edge for edge in edges}
            edge_ids = set(edge_by_id)
            base_edge_ids = set(base_edges)
            union = edge_ids | base_edge_ids
            common = edge_ids & base_edge_ids
            area_changes = [
                abs(float(cell_by_id[key]["area_m2"]) - float(base_cells[key]["area_m2"])) / float(base_cells[key]["area_m2"])
                for key in cell_by_id.keys() & base_cells.keys()
                if float(base_cells[key]["area_m2"]) > 0
            ]
            shared_changes = [
                abs(float(edge_by_id[key]["shared_cell_boundary_m"]) - float(base_edges[key]["shared_cell_boundary_m"])) / float(base_edges[key]["shared_cell_boundary_m"])
                for key in common
                if float(base_edges[key]["shared_cell_boundary_m"]) > 0
            ]
            rows.append(
                {
                    "sample_id": sample_id,
                    "shrink_m": 0.4,
                    "segment_m": segment_m,
                    "sample_point_count": point_count,
                    "runtime_seconds": runtime,
                    "success_cell_count": len(cells),
                    "failed_cell_count": len(failures),
                    "edge_count": len(edges),
                    "edge_jaccard_vs_main": len(common) / len(union) if union else 1.0,
                    "added_edge_count_vs_main": len(edge_ids - base_edge_ids),
                    "removed_edge_count_vs_main": len(base_edge_ids - edge_ids),
                    "cell_area_relative_change_median": float(np.median(area_changes)) if area_changes else 0.0,
                    "cell_area_relative_change_max": max(area_changes, default=0.0),
                    "shared_boundary_relative_change_median": float(np.median(shared_changes)) if shared_changes else 0.0,
                    "shared_boundary_relative_change_max": max(shared_changes, default=0.0),
                    "researcher_judgment_change": "",
                }
            )
            variants.append((segment_m, cells, edges, point_count))

        points = {node["building_id"]: node_point(node) for node in base_graph["nodes"]}
        bounds = extent(block)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
        for ax, (segment_m, cells, edges, point_count) in zip(axes, variants):
            style_axis(ax, bounds)
            draw_cells(ax, cells, alpha=0.38)
            draw_buildings(ax, base_graph["nodes"], alpha=0.72)
            draw_edges(ax, edges, points, linewidth=0.7)
            draw_outline(ax, block)
            ax.set_title(f"segment {segment_m:.1f} m | points {point_count} | edges {len(edges)}")
        fig.suptitle(f"Discretization sensitivity — {sample_id}")
        fig.savefig(SENSITIVITY_DIR / f"{sample_id}_segment_compare.png", dpi=170, facecolor="white")
        plt.close(fig)
        print(f"[sensitivity] {sample_id}", flush=True)
    fields = list(rows[0].keys())
    write_csv(RESULTS_DIR / "parameter_sensitivity.csv", rows, fields)


def decisions(graphs):
    rows = []
    for graph in graphs:
        diagnostics = graph["diagnostics"]
        suggestions = []
        if graph["scope_class"] == "out_of_scope_candidate":
            suggestions.append("confirm study-scope classification")
        if graph["scope_class"] == "data_quality_stress":
            suggestions.append("inspect cross-boundary source geometry")
        if graph["cell_failures"]:
            suggestions.append("inspect empty-after-shrink buildings")
        if diagnostics["long_distance_candidate"]["edge_count"]:
            suggestions.append("inspect highlighted long-distance edges")
        rows.append(
            {
                "sample_id": graph["sample"]["sample_id"],
                "city": graph["graph"]["city"],
                "block_id": graph["graph"]["block_id"],
                "scope_class_auto": graph["scope_class"],
                "validation_passed_auto": diagnostics["validation"]["passed"],
                "failed_cell_count_auto": len(graph["cell_failures"]),
                "long_distance_candidate_count_auto": diagnostics["long_distance_candidate"]["edge_count"],
                "suggested_review": "; ".join(suggestions) or "general visual review",
                "cell_interpretability": "",
                "adjacency_plausibility": "",
                "cross_void_edges": "",
                "relation_value_plausibility": "",
                "parameter_stability": "",
                "sample_scope_decision": "",
                "overall_decision": "",
                "researcher_note": "",
            }
        )
    write_csv(RESULTS_DIR / "visual_review_decisions.csv", rows, list(rows[0].keys()))


def main():
    started = time.perf_counter()
    for directory in (SAMPLE_DIR, REPRESENTATIVE_DIR, SENSITIVITY_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    graphs = prepare_graphs()
    graph_by_id = {graph["sample"]["sample_id"]: graph for graph in graphs}
    sample_paths = []
    for graph in graphs:
        sample_id = graph["sample"]["sample_id"]
        path = SAMPLE_DIR / f"{sample_id}_four_panel.png"
        plot_four_panel(graph, path)
        sample_paths.append(path)
        print(f"[four-panel] {sample_id}", flush=True)
    for sample_id in REPRESENTATIVE_SAMPLES:
        plot_six_panel(
            graph_by_id[sample_id], REPRESENTATIVE_DIR / f"{sample_id}_six_panel.png"
        )
        print(f"[six-panel] {sample_id}", flush=True)
    make_contact_sheet(sample_paths)
    payload = json.dumps(html_payload(graphs), ensure_ascii=False, separators=(",", ":"))
    (RESULTS_DIR / "visual_review.html").write_text(
        HTML_TEMPLATE.replace("__DATA__", payload), encoding="utf-8"
    )
    decisions(graphs)
    sensitivity(graph_by_id)
    print(json.dumps({"visual_samples": len(graphs), "seconds": round(time.perf_counter() - started, 3)}), flush=True)


if __name__ == "__main__":
    main()
