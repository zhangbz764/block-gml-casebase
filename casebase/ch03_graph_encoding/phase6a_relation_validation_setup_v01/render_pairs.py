"""Read selected Graphs and render offline, metrically comparable case cards."""

from __future__ import annotations

import gzip
from html import escape
import json
import math
from pathlib import Path, PureWindowsPath
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.patches import PathPatch
from matplotlib.path import Path as PlotPath
import numpy as np
from shapely import wkt
from shapely.geometry.polygon import orient

from model_inputs import finite_number, source_value
from prepare_pairs import parse_bool


font_path = Path("C:/Windows/Fonts/msyh.ttc")
if font_path.is_file():
    matplotlib.rcParams["font.family"] = FontProperties(fname=str(font_path)).get_name()
matplotlib.rcParams["axes.unicode_minus"] = False
MISSING_HEIGHT_COLOR = "#bdbdbd"
WEAK_COLORS = {"combined_weak_candidate": "#d62728", "long_distance_candidate": "#ff7f0e", "low_shared_candidate": "#9467bd"}
RELATION_GROUPS = (
    ("footprint_gap_m", "建筑轮廓最短间距", "m", ("footprint_gap_m",)),
    ("footprint_gap_block_norm", "间距／街区尺度", "无量纲", ("footprint_gap_block_norm",)),
    ("shared_boundary_ratio_min", "共享Cell边界／较小Cell周长", "比例", ("shared_boundary_ratio_min",)),
    ("orientation_diff_deg", "相邻建筑主轴方向差", "°（0—90）", ("orientation_diff_deg",)),
    ("connection_axis_alignment", "建筑主轴与连线夹角（两端合并）", "°（0—90）", ("connection_axis_alignment_i", "connection_axis_alignment_j")),
    ("height_diff_m", "相邻建筑高度差", "m", ("height_diff_m",)),
    ("log_area_ratio_abs", "建筑轮廓面积对数比绝对值", "无量纲", ("log_area_ratio_abs",)),
)
EXAMPLE_FIELDS = tuple(dict.fromkeys(field for _, _, _, fields in RELATION_GROUPS for field in fields))


def load_graph(index_row: dict, phase4b_root: Path) -> dict:
    root = Path(phase4b_root).resolve()
    raw = str(index_row.get("graph_path", "")).strip()
    if not raw or Path(raw).is_absolute() or PureWindowsPath(raw).drive:
        raise ValueError("Graph path must be relative to the Phase 4B directory")
    path = (root / raw).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError("Graph path escapes the Phase 4B directory") from error
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            record = json.load(handle)
    else:
        with path.open(encoding="utf-8") as handle:
            record = json.load(handle)
    if record.get("encoding_version") != "0.3.2" or not parse_bool(record.get("validation", {}).get("passed")):
        raise ValueError(f"Selected Graph is not a validated v0.3.2 record: {path}")
    for field in ("sample_id", "city", "block_id"):
        if field in index_row and str(record.get("sample", {}).get(field)) != str(index_row[field]):
            raise ValueError(f"Selected Graph identity mismatch: {field}")
    return record


def select_example_edges(record: dict) -> list[dict]:
    edges = sorted(record.get("edges", []), key=lambda item: str(item["edge_id"]))
    gaps = [edge for edge in edges if source_value(edge, "footprint_gap_m") is not None]
    heights = [edge for edge in edges if source_value(edge, "height_diff_m") is not None]
    lists = (
        sorted(gaps, key=lambda edge: (source_value(edge, "footprint_gap_m"), str(edge["edge_id"])))[:2],
        sorted(gaps, key=lambda edge: (-source_value(edge, "footprint_gap_m"), str(edge["edge_id"])))[:2],
        sorted(heights, key=lambda edge: (-source_value(edge, "height_diff_m"), str(edge["edge_id"])))[:2],
    )
    chosen, seen = [], set()
    for group in lists:
        for edge in group:
            if str(edge["edge_id"]) not in seen:
                chosen.append(edge); seen.add(str(edge["edge_id"]))
    return sorted(chosen, key=lambda edge: str(edge["edge_id"]))


def relation_summary(record: dict) -> list[dict]:
    edges = record.get("edges", [])
    rows = []
    for key, label, unit, fields in RELATION_GROUPS:
        values, valid_edges = [], 0
        for edge in edges:
            row = [source_value(edge, field) for field in fields]
            valid = [value for value in row if value is not None]
            values.extend(valid); valid_edges += bool(valid)
        q1, median, q3 = np.quantile(values, (0.25, 0.5, 0.75)) if values else (None, None, None)
        rows.append({"relation": key, "label": label, "unit": unit, "edge_count": len(edges),
                     "valid_edges": int(valid_edges), "valid_values": len(values),
                     "q1": float(q1) if q1 is not None else None,
                     "median": float(median) if median is not None else None,
                     "q3": float(q3) if q3 is not None else None,
                     "iqr": float(q3 - q1) if q1 is not None else None})
    return rows


def _geometry(value):
    return wkt.loads(value) if value else None


def _polygons(geometry):
    if geometry is None or geometry.is_empty:
        return []
    return list(geometry.geoms) if geometry.geom_type == "MultiPolygon" else [geometry]


def _draw_geometry(axis, geometry, color, face="none", linewidth=0.8, alpha=1.0):
    # Extends Phase 5's polygon/ring drawing; preserves holes and source orientation.
    for polygon in _polygons(geometry):
        polygon = orient(polygon, sign=1.0)
        vertices, codes = [], []
        for ring in (polygon.exterior, *polygon.interiors):
            coordinates = list(ring.coords)
            vertices.extend(coordinates)
            codes.extend([PlotPath.MOVETO, *([PlotPath.LINETO] * (len(coordinates) - 2)), PlotPath.CLOSEPOLY])
        axis.add_patch(PathPatch(PlotPath(vertices, codes), facecolor=face, edgecolor=color,
                                 linewidth=linewidth, alpha=alpha))


def draw_pair_view(record_a: dict, record_b: dict, *, mode="plan", show_cells=False):
    if mode not in ("plan", "plain", "weak"):
        raise ValueError("Unknown drawing mode")
    records = (record_a, record_b)
    geometries = []
    bounds = []
    heights = []
    for record in records:
        block = _geometry(record["graph"]["block_geometry_wkt_metric"])
        buildings = [_geometry(node.get("geometry_wkt_metric")) for node in record.get("nodes", [])]
        cells = [_geometry(cell.get("geometry_wkt_metric")) for cell in record.get("cells", [])]
        visible = [geometry for geometry in (block, *buildings) if geometry is not None and not geometry.is_empty]
        bounds.append((min(g.bounds[0] for g in visible), min(g.bounds[1] for g in visible),
                       max(g.bounds[2] for g in visible), max(g.bounds[3] for g in visible)))
        geometries.append((block, buildings, cells))
        heights.extend(value for node in record.get("nodes", []) if (value := source_value(node, "height_m")) is not None)
    span = max(1.0, *(max(box[2] - box[0], box[3] - box[1]) for box in bounds)) * 1.12
    height_min = min(heights) if heights else None
    height_max = max(heights) if heights else None
    norm = Normalize(height_min or 0.0, max(height_max or 1.0, (height_min or 0.0) + 1e-6))
    cmap = plt.get_cmap("viridis")
    figure = plt.figure(figsize=(12, 5.8), dpi=130)
    axes = [figure.add_axes((0.045, 0.2, 0.43, 0.65)), figure.add_axes((0.525, 0.2, 0.43, 0.65))]
    for side, (record, axis, (block, buildings, cells), box) in enumerate(zip(records, axes, geometries, bounds, strict=True)):
        center_x, center_y = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        axis.set_xlim(center_x - span / 2, center_x + span / 2)
        axis.set_ylim(center_y - span / 2, center_y + span / 2)
        axis.set_aspect("equal", adjustable="box"); axis.axis("off")
        _draw_geometry(axis, block, "#222222", linewidth=1.2)
        for node, geometry in zip(record.get("nodes", []), buildings, strict=True):
            height = source_value(node, "height_m")
            face = MISSING_HEIGHT_COLOR if height is None else cmap(norm(height))
            _draw_geometry(axis, geometry, "#555555" if mode == "plan" else "#c7c7c7",
                           face=face if mode == "plan" else "#f0f0f0", linewidth=0.5, alpha=1.0)
        if show_cells:
            for geometry in cells:
                _draw_geometry(axis, geometry, "#76975e", linewidth=0.55, alpha=0.75)
        centers = {str(node["building_id"]): (finite_number(node.get("representative_point_x_m")), finite_number(node.get("representative_point_y_m")))
                   for node in record.get("nodes", [])}
        examples = select_example_edges(record)
        ranks = {str(edge["edge_id"]): i for i, edge in enumerate(examples, 1)}
        if mode != "plan":
            for edge in record.get("edges", []):
                start, end = centers[str(edge["source"])], centers[str(edge["target"])]
                if any(value is None for value in (*start, *end)):
                    raise ValueError("An edge lacks recorded metric endpoint coordinates")
                color = "#777777"
                if mode == "weak":
                    color = next((color for flag, color in WEAK_COLORS.items() if parse_bool(edge.get(flag))), "#b0b0b0")
                axis.plot((start[0], end[0]), (start[1], end[1]), color=color, linewidth=0.9, zorder=4)
                rank = ranks.get(str(edge["edge_id"]))
                if rank is not None:
                    axis.plot((start[0], end[0]), (start[1], end[1]), color=color, linewidth=2.0, zorder=5)
                    axis.text((start[0] + end[0]) / 2, (start[1] + end[1]) / 2, f"{'AB'[side]}{rank}", fontsize=8,
                              bbox={"facecolor": "white", "edgecolor": "#444", "pad": 1.2}, zorder=6)
        axis.set_title(f"{'AB'[side]} | {record.get('sample', {}).get('block_id', '')} | {len(record.get('nodes', []))} buildings", fontsize=10)
        axis.plot((center_x - span * 0.4, center_x - span * 0.2), (center_y - span * 0.43,) * 2, color="black", linewidth=2)
        axis.text(center_x - span * 0.3, center_y - span * 0.47, f"{span * 0.2:.1f} m", ha="center", fontsize=8)
    if mode == "plan" and heights:
        color_axis = figure.add_axes((0.25, 0.085, 0.5, 0.028))
        figure.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=color_axis, orientation="horizontal", label="Building height (m); grey = missing")
    elif mode == "weak":
        labels = (("#b0b0b0", "ordinary"), ("#ff7f0e", "long distance"), ("#9467bd", "low shared boundary"), ("#d62728", "combined"))
        figure.legend([Line2D([0], [0], color=color, lw=2) for color, _ in labels], [label for _, label in labels],
                      loc="lower center", ncol=4, title="Weak-edge flags: not errors, not deletions", fontsize=8)
    elif mode == "plain":
        figure.legend([Line2D([0], [0], color="#777777", lw=2)], ["Complete adjacency; numbered example edges"],
                      loc="lower center", fontsize=8)
    else:
        figure.text(0.5, 0.09, "Height unavailable; all footprints grey", ha="center")
    figure.canvas.draw()
    scales = [span / axis.get_window_extent().width for axis in axes]
    if not math.isclose(scales[0], scales[1], rel_tol=1e-9):
        plt.close(figure)
        raise ValueError("A/B drawing has inconsistent metres per pixel")
    info = {"world_span_m": span, "meters_per_pixel": scales, "height_min_m": height_min,
            "height_max_m": height_max, "missing_height_color": MISSING_HEIGHT_COLOR, "equal_aspect": True}
    return figure, axes, info


def render_pair(record_a: dict, record_b: dict, pair_row: dict, output_dir: Path) -> dict:
    pair_id = str(pair_row["pair_id"])
    if not re.fullmatch(r"[A-Za-z0-9_-]+", pair_id):
        raise ValueError("Unsafe pair ID")
    image_dir = Path(output_dir) / "images"; image_dir.mkdir(parents=True, exist_ok=True)
    images, checks = {}, []
    for name, mode, cells in (("plan", "plan", False), ("plain", "plain", False), ("plain_cells", "plain", True),
                              ("weak", "weak", False), ("weak_cells", "weak", True)):
        path = image_dir / f"{pair_id}_{name}.png"
        if path.exists():
            raise FileExistsError(path)
        figure, _, info = draw_pair_view(record_a, record_b, mode=mode, show_cells=cells)
        try:
            figure.savefig(path, dpi=130)
        finally:
            plt.close(figure)
        images[name] = path.relative_to(output_dir).as_posix(); checks.append(info)
    rows, example_rows, summaries = [], [], {}
    counts = {}
    for side, record in (("a", record_a), ("b", record_b)):
        summaries[side] = relation_summary(record)
        rows.extend({"pair_id": pair_id, "side": side.upper(), "sample_id": record.get("sample", {}).get("sample_id"), **row} for row in summaries[side])
        values = record.get("block_attributes", {}).get("values", {})
        counts[side] = {"node_count": len(record.get("nodes", [])), "lod1_building_count": values.get("lod1_building_count"),
                        "excluded_building_count": record.get("graph", {}).get("excluded_building_count"),
                        "cell_failure_count": len(record.get("cell_failures", [])), "edge_count": len(record.get("edges", []))}
        for rank, edge in enumerate(select_example_edges(record), 1):
            missing = {field: edge.get("missing_reasons", {}).get(field, "source_missing")
                       for field in EXAMPLE_FIELDS if source_value(edge, field) is None}
            example_rows.append({"pair_id": pair_id, "side": side.upper(), "sample_id": record.get("sample", {}).get("sample_id"), "label": f"{side.upper()}{rank}",
                                 "edge_id": str(edge["edge_id"]), "source": str(edge["source"]), "target": str(edge["target"]),
                                 **{field: source_value(edge, field) for field in EXAMPLE_FIELDS},
                                 "missing_reasons": json.dumps(missing, ensure_ascii=False)})
    return {"pair_id": pair_id, "images": images, "checks": checks, "relation_rows": rows,
            "summaries": summaries, "example_edge_rows": example_rows, "counts": counts}


def metric_rows(pair: dict, spec: dict) -> list[dict]:
    rows = []
    active = str(pair["fields_used"]).split(";")
    for field, definition in spec["pair_fields"].items():
        a, b = finite_number(pair.get(f"{field}_a")), finite_number(pair.get(f"{field}_b"))
        rows.append({"pair_id": pair["pair_id"], "city": pair["city"], "field": field,
                     "label": definition.get("label", field), "unit": definition.get("unit", ""),
                     "a_value": a, "b_value": b, "absolute_difference": abs(a-b) if a is not None and b is not None else None,
                     "standardized_difference": pair.get(f"std_diff_{field}"), "used_in_distance": field in active})
    return rows


def _value(value):
    number = finite_number(value)
    return "—" if number is None else f"{number:.5g}"


def build_review_html(pairs, cards: list[dict], html_path: Path, spec: dict, summary: dict) -> None:
    articles = []
    for pair, card in zip(pairs.to_dict("records"), cards, strict=True):
        metric_table = "".join(f"<tr><th>{escape(row['label'])} ({escape(row['unit'])})</th><td>{_value(row['a_value'])}</td><td>{_value(row['b_value'])}</td><td>{_value(row['absolute_difference'])}</td><td>{_value(row['standardized_difference']) if row['used_in_distance'] else '未参与'}</td></tr>" for row in metric_rows(pair, spec))
        relation_table = "".join(f"<tr><th>{escape(a['label'])} ({escape(a['unit'])})</th>" + "".join(
            f"<td>{_value(row['median'])} / {_value(row['iqr'])}；有效边 {row['valid_edges']}/{row['edge_count']}，值 {row['valid_values']}</td>" for row in (a,b)) + "</tr>"
            for a,b in zip(card["summaries"]["a"], card["summaries"]["b"], strict=True))
        edges = "".join(f"<tr><th>{escape(row['label'])}: {escape(row['source'])} → {escape(row['target'])}</th><td>" +
                        "；".join(f"{field}={_value(row[field])}" for field in EXAMPLE_FIELDS) +
                        f"<br>缺失原因：{escape(row['missing_reasons'])}</td></tr>" for row in card["example_edge_rows"])
        warning = "<p class='warning'>nearest_only：不足以直接视为指标接近。</p>" if pair["candidate_level"] == "nearest_only" else ""
        count_text = "；".join(f"{side.upper()}：源建筑数 {_value(c['lod1_building_count'])}，编码建筑数 {c['node_count']}，排除 {_value(c['excluded_building_count'])}，Cell failure {c['cell_failure_count']}" for side,c in card["counts"].items())
        no_edges = "".join(f"<p>{side.upper()}：此Graph仅有节点，无关系统计。</p>" for side,c in card["counts"].items() if c["edge_count"] == 0)
        images = escape(json.dumps(card["images"]), quote=True)
        articles.append(f"""<article data-pair="{escape(pair['pair_id'])}" data-images="{images}">
<h2>{escape(pair['pair_id'])} · {escape(pair['city'])} · {escape(pair['candidate_level'])}</h2>{warning}
<p>A：{escape(str(pair['block_id_a']))} / {escape(str(pair['sample_id_a']))}；B：{escape(str(pair['block_id_b']))} / {escape(str(pair['sample_id_b']))}</p>
<p>平均标准化差值 {_value(pair['mean_standardized_difference'])}；最大单字段差值 {_value(pair['max_standardized_difference'])}。{escape(count_text)}</p>
<table><tr><th>指标</th><th>A原值</th><th>B原值</th><th>原值绝对差</th><th>城市IQR标准化差</th></tr>{metric_table}</table>
<h3>总平面：同一米/像素尺度、共同高度色标、保持源方向</h3><img loading="lazy" src="{escape(card['images']['plan'])}" alt="A/B总平面">
<h3>完整邻接与关系标记</h3><label>边图 <select class="edge-mode"><option value="plain">普通边</option><option value="weak">弱边标记</option></select></label>
<label><input type="checkbox" class="cells">显示Cell边界</label><p>灰：普通；橙：长距离候选；紫：低共享边界候选；红：组合弱边。弱边标记≠错误≠删除。缺失Cell不删除建筑。</p>
<img class="relation-image" loading="lazy" src="{escape(card['images']['plain'])}" alt="A/B关系图">{no_edges}
<h3>已有关系属性：中位数 / IQR</h3><table><tr><th>解释量</th><th>A</th><th>B</th></tr>{relation_table}</table>
<p>共享的是Cell边界，不是建筑立面；主轴与连线夹角不直接称作面对关系。分布摘要仅用于解释，不是latent space或新增模型特征。</p>
<details><summary>编号示例边：每图最多6条，短间距／长间距／大高度差</summary><table>{edges}</table></details>
<details><summary>研究者判断（可只选约8—12对并给一段总判断）</summary>
<label>总体指标是否足够接近？<input data-field="metrics_close_enough"></label>
<label>实际空间组织有何差异？<textarea data-field="layout_difference"></textarea></label>
<label>图中能否找到对应关系？<textarea data-field="encoding_correspondence"></textarea></label>
<label>是否用于正文？<input data-field="keep_for_thesis"></label><label>备注<textarea data-field="comment"></textarea></label>
</details></article>""")
    total = len(cards)
    document = """<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>Phase 6A 形态对照候选</title>
<style>body{font-family:'Microsoft YaHei',system-ui,sans-serif;max-width:1280px;margin:2rem auto;padding:0 1rem;color:#222}article{border-top:2px solid #aaa;margin-top:3rem;padding-top:1rem}table{border-collapse:collapse;width:100%;font-size:.9rem}th,td{border:1px solid #ddd;padding:.5rem;text-align:left}img{width:100%;height:auto}label{display:block;margin:.7rem 0}input,textarea{display:block;max-width:95%;width:48rem;padding:.4rem}input[type=checkbox]{display:inline;width:auto}textarea{min-height:3rem}.warning{background:#fff0ce;padding:1rem}details{margin:1rem 0}button,select{padding:.4rem}summary{cursor:pointer}</style></head><body>
<h1>指标接近候选与关系编码对照</h1>""" + f"<p>本轮 {total} 对，{len(set(pairs['city'])) if total else 0} 城；未训练模型，候选不自动等于布局不同，也不证明图模型优越。</p>"
    document += "<p>close/relaxed界限为本阶段操作参数，不是相同形态判据。总体指标沿用上游源口径，可能包含后来排除的异常建筑；编码建筑数是有效Node数，平均高度为面积加权绝对高度。</p><p>页面输入不会自动保存，请填写后点击导出。也可直接向执行工作区提供选定pair_id与一段研究判断。</p><button id='export-review'>导出人工判断CSV</button>"
    if not total:
        document += "<p class='warning'>没有入选候选。按实际数量交付，没有放宽规则或插补指标；原因见sampling_summary.json。</p>"
    document += "".join(articles)
    document += r"""<script>
document.querySelectorAll('article').forEach(card=>{const imgs=JSON.parse(card.dataset.images);function update(){const key=card.querySelector('.edge-mode').value+(card.querySelector('.cells').checked?'_cells':'');card.querySelector('.relation-image').src=imgs[key]}card.querySelector('.edge-mode').onchange=update;card.querySelector('.cells').onchange=update});
document.getElementById('export-review').onclick=()=>{const fields=['pair_id','metrics_close_enough','layout_difference','encoding_correspondence','keep_for_thesis','comment'];const quote=v=>'"'+String(v).replaceAll('"','""')+'"';const rows=[fields,...Array.from(document.querySelectorAll('article')).map(card=>fields.map(f=>f==='pair_id'?card.dataset.pair:card.querySelector('[data-field="'+f+'"]').value))];const blob=new Blob(['\ufeff'+rows.map(row=>row.map(quote).join(',')).join('\r\n')],{type:'text/csv;charset=utf-8'});const url=URL.createObjectURL(blob);const a=document.createElement('a');a.href=url;a.download='pair_review.csv';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)};
</script></body></html>"""
    html_path.write_text(document, encoding="utf-8")
