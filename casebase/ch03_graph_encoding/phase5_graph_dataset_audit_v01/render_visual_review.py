"""Render local, equal-aspect visual review panels for selected Graphs."""

from __future__ import annotations

from html import escape
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from shapely import wkt


def _draw_geometry(axis, value: str | None, color: str, linewidth: float = 0.8, alpha: float = 1.0) -> None:
    if not value:
        return
    geometry = wkt.loads(value)
    polygons = list(geometry.geoms) if geometry.geom_type == "MultiPolygon" else [geometry]
    for polygon in polygons:
        x, y = polygon.exterior.xy
        axis.plot(x, y, color=color, linewidth=linewidth, alpha=alpha)
        for ring in polygon.interiors:
            x, y = ring.xy; axis.plot(x, y, color=color, linewidth=linewidth * 0.7, alpha=alpha)


def _centres(record: dict) -> dict[str, tuple[float, float]]:
    return {str(node["building_id"]): (float(node["representative_point_x_m"]), float(node["representative_point_y_m"])) for node in record.get("nodes", [])}


def render_review_case(record: dict, output_path: Path, metadata: dict) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    graph = record.get("graph", {}); centres = _centres(record)
    for axis in axes:
        _draw_geometry(axis, graph.get("block_geometry_wkt_metric"), "#222222", 1.4)
        axis.set_aspect("equal", adjustable="box"); axis.axis("off")
    for node in record.get("nodes", []):
        _draw_geometry(axes[0], node.get("geometry_wkt_metric"), "#2878b5", 0.9)
    axes[0].set_title("Building footprints")
    for cell in record.get("cells", []):
        _draw_geometry(axes[1], cell.get("geometry_wkt_metric"), "#6a9f58", 0.7)
    for edge in record.get("edges", []):
        source, target = centres.get(str(edge.get("source"))), centres.get(str(edge.get("target")))
        if source and target:
            axes[1].plot([source[0], target[0]], [source[1], target[1]], color="#777777", linewidth=0.7)
    axes[1].set_title("Cells and final edges")
    colours = {"combined_weak_candidate": "#d62728", "long_distance_candidate": "#ff7f0e", "low_shared_candidate": "#9467bd"}
    for edge in record.get("edges", []):
        source, target = centres.get(str(edge.get("source"))), centres.get(str(edge.get("target")))
        if not source or not target:
            continue
        flags = [name for name in colours if edge.get(name)]
        color = colours[flags[0]] if flags else "#cccccc"
        axes[2].plot([source[0], target[0]], [source[1], target[1]], color=color, linewidth=1.6 if flags else 0.5)
    axes[2].set_title("Weak-edge review (not deletion)")
    title = (f"{metadata.get('city')} | {metadata.get('block_id')} | nodes={metadata.get('node_count')} "
             f"edges={metadata.get('edge_count')} components={metadata.get('connected_components')} "
             f"weak={metadata.get('combined_weak_candidate_count')} excluded={metadata.get('excluded_building_count')}")
    figure.suptitle(title, fontsize=10)
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def build_review_html(manifest: pd.DataFrame, html_path: Path) -> None:
    html_path.parent.mkdir(parents=True, exist_ok=True)
    options = "".join(f'<option value="{escape(str(city))}">{escape(str(city))}</option>' for city in sorted(manifest["city"].unique()))
    cards = []
    for row in manifest.to_dict("records"):
        image = escape(str(row.get("image_path", "")))
        cards.append(
            f'<article data-city="{escape(str(row["city"]))}" data-source="{escape(str(row["selection_source"]))}">'
            f'<h3>{escape(str(row["city"]))} — {escape(str(row["block_id"]))}</h3>'
            f'<p>{escape(str(row["selection_source"]))}; nodes={row.get("node_count")}; edges={row.get("edge_count")}; '
            f'components={row.get("connected_components")}</p><img src="{image}" loading="lazy"></article>'
        )
    document = f"""<!doctype html><meta charset="utf-8"><title>Phase 5 visual review</title>
<style>body{{font-family:system-ui;margin:2rem}}img{{max-width:100%}}article{{margin:2rem 0}}label{{margin-right:1rem}}</style>
<label>City <select id="city"><option value="">All</option>{options}</select></label>
<label>Source <select id="source"><option value="">All</option><option>city_median</option><option>city_structural_extreme</option></select></label>
{''.join(cards)}<script>function f(){{let c=city.value,s=source.value;document.querySelectorAll('article').forEach(x=>x.hidden=(c&&x.dataset.city!==c)||(s&&x.dataset.source!==s))}}city.onchange=f;source.onchange=f;</script>"""
    html_path.write_text(document, encoding="utf-8")


def write_manual_review(manifest: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"sample_id": manifest["sample_id"], "visual_status": "", "issue_type": "", "comment": ""}).to_csv(
        path, index=False, encoding="utf-8-sig", lineterminator="\n"
    )
