"""A short offline probe page; no controls, fabricated after-images or model claims."""

from __future__ import annotations

from html import escape
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from shapely import wkt

from control_probes import EXAMPLE_PAIRS, FIELDS, GROUPS, TARGET_FIELDS, edge_comparisons, source_value, summarize_probe
from render_pairs import _draw_geometry, WEAK_COLORS
from prepare_pairs import parse_bool

LABELS = {
    "footprint_gap_m": "轮廓间距 m", "footprint_gap_block_norm": "间距/街区尺度",
    "centroid_distance_block_norm": "中心距离/街区尺度", "long_side_overlap_ratio": "长边投影重叠比例",
    "long_side_offset_norm": "主轴法向间隔（归一化）", "connection_axis_alignment_i": "source主轴与连线夹角 °",
    "connection_axis_alignment_j": "target主轴与连线夹角 °", "orientation_diff_deg": "主轴方向差 °",
    "height_diff_m": "高度差 m", "log_height_ratio_abs": "高度对数比绝对值",
}
OPERATIONS = {
    "gap": "中心相对(60,40)缩为0.8倍、保持基准、扩为1.2倍；轮廓尺寸及方向不变。",
    "offset": "下排向左、上排向右，分别形成8m、16m相对横向错位；方向和y坐标不变。",
    "height": "两例都保留六栋8m、两栋24m，高层ID由B01/B02改为B01/B08；平面几何相同。",
}
OPERABLE = {
    "gap": "指定某些相邻建筑对的目标间隔或紧疏方向，同时保留轮廓、场地边界和不相交约束。",
    "offset": "指定两排相对横向位移，不把对齐角直接当作立面面对关系；需生成器可操纵建筑位置。",
    "height": "保持高度集合及平面条件，调整高体量分配给哪些建筑ID，并核对具体高低邻接。",
}


def number(value) -> str:
    return "缺失" if value is None else f"{value:.5g}"


def draw_plan_panels(records: list, *, synthetic: bool, target: str = "gap", example_pairs=None):
    count = len(records)
    if count not in (1, 2, 3):
        raise ValueError("Use one to three panels per card")
    heights = [value for record in records if record for node in record.get("nodes", [])
               if (value := source_value(node, "height_m")) is not None]
    height_range = [8.0, 24.0] if synthetic else ([min(heights), max(heights)] if heights else [None, None])
    norm = Normalize(height_range[0] if height_range[0] is not None else 0,
                     max(height_range[1], height_range[0]+1e-6) if height_range[1] is not None else 1)
    cmap = plt.get_cmap("viridis")
    figure = plt.figure(figsize=(4.5*count, 5), dpi=140)
    axes = [figure.add_axes((i/count+0.025, 0.23, 1/count-0.05, 0.64)) for i in range(count)]
    ranges = []
    for axis, record in zip(axes, records, strict=True):
        if synthetic:
            xlim, ylim = (-5, 125), (-5, 85)
        elif record:
            geometries = [wkt.loads(record["graph"]["block_geometry_wkt_metric"])]
            geometries.extend(wkt.loads(n["geometry_wkt_metric"]) for n in record.get("nodes", []))
            left = min(g.bounds[0] for g in geometries); right = max(g.bounds[2] for g in geometries)
            bottom = min(g.bounds[1] for g in geometries); top = max(g.bounds[3] for g in geometries)
            span = max(right-left, top-bottom, 1)*1.12
            xlim = ((left+right-span)/2, (left+right+span)/2)
            ylim = ((bottom+top-span)/2, (bottom+top+span)/2)
        else:
            xlim, ylim = (0, 1), (0, 1)
        axis.set_xlim(*xlim); axis.set_ylim(*ylim); axis.set_aspect("equal", adjustable="box")
        ranges.append([list(xlim), list(ylim)])
        axis.axis("off")
        if record is None:
            axis.text(0.5, 0.5, "未完成编码，见失败记录", transform=axis.transAxes, ha="center")
            continue
        _draw_geometry(axis, wkt.loads(record["graph"]["block_geometry_wkt_metric"]), "#222", linewidth=1.3)
        centres = {}
        for node in record.get("nodes", []):
            height = source_value(node, "height_m")
            _draw_geometry(axis, wkt.loads(node["geometry_wkt_metric"]), "#555",
                           face="#bdbdbd" if height is None else cmap(norm(height)), linewidth=0.5)
            centres[str(node["building_id"])] = (float(node["representative_point_x_m"]), float(node["representative_point_y_m"]))
            if synthetic:
                axis.text(*centres[str(node["building_id"])], str(node["building_id"]), fontsize=8, ha="center", va="center",
                          bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 0.8}, zorder=8)
        example_keys = [frozenset(pair) for pair in (example_pairs if example_pairs is not None else EXAMPLE_PAIRS[target])]
        for edge in record.get("edges", []):
            start, end = centres[str(edge["source"])], centres[str(edge["target"])]
            color = next((color for flag,color in WEAK_COLORS.items() if parse_bool(edge.get(flag))), "#999999")
            axis.plot((start[0], end[0]), (start[1], end[1]), color=color, linewidth=0.65, alpha=0.75, zorder=3)
            key = frozenset((str(edge["source"]), str(edge["target"])))
            if key in example_keys:
                rank = example_keys.index(key)+1
                axis.plot((start[0], end[0]), (start[1], end[1]), color="#0072b2", linewidth=1.8, zorder=4)
                axis.annotate(f"E{rank}", ((start[0]+end[0])/2, (start[1]+end[1])/2),
                              xytext=(3, 6*rank), textcoords="offset points", fontsize=8,
                              bbox={"facecolor": "white", "edgecolor": "#0072b2", "pad": 0.8}, zorder=9)
        sample_id = str(record.get("sample", {}).get("sample_id", ""))
        axis.set_title(sample_id, fontsize=10)
        if synthetic and target == "offset" and sample_id.startswith("OFFSET_"):
            shift = 4 if sample_id == "OFFSET_08" else 8
            for y, amount in ((30, -shift), (50, shift)):
                axis.annotate("", xy=(60+amount, y), xytext=(60, y), arrowprops={"arrowstyle": "->", "color": "#c21"})
        width = xlim[1]-xlim[0]
        x, y = xlim[0]+width*0.07, ylim[0]+(ylim[1]-ylim[0])*0.04
        axis.plot([x, x+width*0.2], [y, y], color="black", linewidth=2)
        axis.text(x+width*0.1, y, f"{width*0.2:.1f} m", ha="center", va="top", fontsize=8)
    if heights or synthetic:
        color_axis = figure.add_axes((0.25, 0.10, 0.5, 0.025))
        figure.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=color_axis,
                        orientation="horizontal", label=f"Height (m): {number(height_range[0])}–{number(height_range[1])}; grey = missing")
    else:
        figure.text(0.5, 0.10, "Height unavailable; grey footprints", ha="center")
    figure.text(0.5, 0.015, "Grey/orange/purple/red: ordinary/long/low shared/combined; blue E1/E2: examples. Weak flags ≠ deletion.", ha="center", fontsize=8)
    figure.canvas.draw()
    scales = [(ranges[i][0][1]-ranges[i][0][0])/axis.get_window_extent().width for i,axis in enumerate(axes)]
    if synthetic and any(not math.isclose(scale, scales[0], rel_tol=1e-9) for scale in scales):
        plt.close(figure)
        raise ValueError("Synthetic panels have inconsistent metres per pixel")
    return figure, axes, {"axis_ranges": ranges, "meters_per_pixel": scales, "height_range_m": height_range, "equal_aspect": True}


def draw_height_axons(records: list):
    figure = plt.figure(figsize=(11, 5.5), dpi=140)
    norm = Normalize(8, 24); cmap = plt.get_cmap("viridis")
    for index, record in enumerate(records, 1):
        axis = figure.add_subplot(1, 2, index, projection="3d")
        axis.set_xlim(-5, 125); axis.set_ylim(-5, 85); axis.set_zlim(0, 30)
        axis.set_box_aspect((130, 90, 30)); axis.view_init(elev=28, azim=-60)
        axis.set_xlabel("x (m)"); axis.set_ylabel("y (m)"); axis.set_zlabel("Height (m)")
        if record is None:
            axis.set_title("未完成编码"); continue
        for node in record["nodes"]:
            x0,y0,x1,y1 = wkt.loads(node["geometry_wkt_metric"]).bounds
            height = source_value(node, "height_m")
            if height is None:
                continue
            axis.bar3d(x0, y0, 0, x1-x0, y1-y0, height, color=cmap(norm(height)), shade=True, linewidth=0.25, edgecolor="#555")
        axis.set_title(record["sample"]["sample_id"]+" | schematic prisms")
    color_axis = figure.add_axes((0.27, 0.06, 0.46, 0.025))
    figure.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=color_axis, orientation="horizontal", label="Height (m): same 8–24 scale")
    return figure


def describe_response(records: dict, target: str) -> str:
    if not records:
        return "没有完成编码，不能判断响应。"
    summaries = {name: summarize_probe(record) for name,record in records.items()}
    descriptions = []
    for field in TARGET_FIELDS[target]:
        values = [row[f"{field}_median"] for row in summaries.values()]
        prefix = f"{LABELS[field]}中位数："+" → ".join(number(value) for value in values)
        valid = [value for value in values if value is not None]
        if not valid:
            prefix += "（没有有效记录）"
        elif len(valid) == len(values) and all(math.isclose(value,valid[0],rel_tol=1e-9,abs_tol=1e-9) for value in valid):
            prefix += "（摘要未变，须看端点和分布）"
        descriptions.append(prefix)
    return "；".join(descriptions)+"。邻接可变化；摘要变化或恒定都不自动判作成功/失败。"


def summary_table(records: dict, target: str) -> str:
    headings = "".join(f"<th>{escape(name)}</th>" for name in records)
    summaries = [summarize_probe(record) for record in records.values()]
    rows = []
    for field in TARGET_FIELDS[target]:
        rows.append(f"<tr><th>{escape(LABELS[field])}</th>"+"".join(
            f"<td>{number(row[field+'_median'])} / {number(row[field+'_iqr'])}；n={row[field+'_valid_count']}</td>" for row in summaries)+"</tr>")
    return f"<table><tr><th>中位数 / IQR；有效数量</th>{headings}</tr>{''.join(rows)}</table>"


def example_table(rows: list, target: str) -> str:
    headings = "".join(f"<th>{escape(LABELS[f])}</th>" for f in TARGET_FIELDS[target])
    body = []
    for row in rows:
        body.append(f"<tr><th>{escape(row['layout_id'])}：{escape(row['endpoint_a'])}—{escape(row['endpoint_b'])}</th>"
                    f"<td>{escape(row['adjacency_note'])}</td>"+"".join(f"<td>{number(row[f])}</td>" for f in TARGET_FIELDS[target])+
                    f"<td>{escape(row['missing_reasons'])}</td></tr>")
    return f"<table><tr><th>同一端点对（蓝色E1/E2）</th><th>邻接</th>{headings}<th>缺失原因</th></tr>{''.join(body)}</table>"


def render_preview(records: dict, real_cards: dict, review_dir: Path) -> dict:
    image_dir = review_dir / "images"; image_dir.mkdir()
    cards = []; checks = {}
    for target, title in (("gap", "一、建筑间隔"), ("offset", "二、两排行列错位"), ("height", "三、高层体量布置")):
        names = GROUPS[target]
        subset = {name: records[name] for name in names if name in records}
        figure, _, check = draw_plan_panels([records.get(name) for name in names], synthetic=True, target=target)
        try:
            figure.savefig(image_dir / f"{target}.png")
        finally:
            plt.close(figure)
        checks[target] = check
        extra = ""
        if target == "height":
            figure = draw_height_axons([records.get(name) for name in names])
            try:
                figure.savefig(image_dir / "height_axons.png")
            finally:
                plt.close(figure)
            extra = '<img src="images/height_axons.png" alt="同一高度集合的八个示意棱柱"><p>共同高度集合：8m×6、24m×2；面积加权平均12m。轴测仅展示位置，不是生成器输出。</p>'
        rows = edge_comparisons(subset, target)
        cards.append(f"<section><h2>{title}</h2><p>人为改变了什么：{OPERATIONS[target]}</p>"
                     f"<img src='images/{target}.png' alt='{title}'>{extra}<p>哪些记录响应/未响应：{escape(describe_response(subset,target))}</p>"
                     f"{summary_table(subset,target)}{example_table(rows,target)}<p>未来设计者可提出什么操作：{OPERABLE[target]}</p></section>")
    real_sections = []
    for target, card in real_cards.items():
        title = {"gap": "建筑间隔", "offset": "错位关系记录", "height": "高度邻接"}[target]
        record = card["record"]
        if record is None:
            real_sections.append(f"<h3>{title}</h3><p>{escape(card['reason'])}</p>"); continue
        figure, _, check = draw_plan_panels([record], synthetic=False, target=target,
                                           example_pairs=[(e["source"],e["target"]) for e in card["examples"]])
        try:
            figure.savefig(image_dir / f"real_{target}.png")
        finally:
            plt.close(figure)
        checks[f"real_{target}"] = check
        examples = [{"layout_id": card["source"]["sample_id"], "endpoint_a": str(e["source"]), "endpoint_b": str(e["target"]),
                     "adjacency_note": "已有记录，未实施移动或生成", **{f: source_value(e,f) for f in FIELDS},
                     "missing_reasons": str(e.get("missing_reasons", {}))} for e in card["examples"]]
        real_sections.append(f"<h3>{title}：{escape(card['source']['city'])} / {escape(card['source']['block_id'])}</h3>"
                             f"<img src='images/real_{target}.png' alt='真实记录'><p>人为改变了什么：没有改变真实案例，仅读取已有记录。</p>"
                             f"<p>哪些记录可读：蓝色E1/E2端点和下表数值。{escape(card['reason'])}；不能由此认定组团、两排行列、立面面对、功能或楼层。</p>"
                             f"{example_table(examples,target)}<p>未来可提出的操作：{OPERABLE[target]}这里只提出操作，没有虚构生成后图。</p>")
    document = """<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>6B前置：控制目标与编码响应</title>
<style>body{font-family:'Microsoft YaHei',system-ui,sans-serif;max-width:1300px;margin:2rem auto;padding:0 1rem}section{border-top:2px solid #bbb;margin-top:2rem}img{width:100%;height:auto}table{border-collapse:collapse;width:100%;font-size:.85rem;margin:1rem 0}th,td{border:1px solid #ddd;padding:.4rem;text-align:left;overflow-wrap:anywhere}p{line-height:1.65}</style></head><body>
<h1>形态控制目标与编码响应：探索性准备</h1><p>七个指定本地布局由程序直接构造，不是模型或完整生成方法的结果。无训练、无重新抽样、无正式图改写。数值只说明局部记录，不证明图不可替代或设计者主体性提高。</p>
<p>示意平面共用坐标范围、米/像素与8—24m色标。灰/橙/紫/红是弱边标记（不删除），蓝色为两条示例。真实图使用自身高度范围、源方向与equal比例。主轴方向差不必随横向错位改变，long_side_offset_norm也不是横向错位量。</p>"""
    document += "".join(cards)+"<section><h2>四、6A已展示真实案例</h2><p>仅提供可读记录，非盲测或生成结果；没有清楚案例就保留缺口。</p>"+"".join(real_sections)+"</section><p>研究者可直接回复：这些变化是否对应想控制的组织？哪个目标值得先做？哪些解释过度？无需填写新表。</p></body></html>"
    (review_dir / "visual_review.html").write_text(document, encoding="utf-8")
    return checks
