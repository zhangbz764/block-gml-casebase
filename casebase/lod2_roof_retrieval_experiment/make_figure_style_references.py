"""Create two editable style references for the conference abstract figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent
QUERY_FIGURE_DIR = (
    ROOT
    / "outputs"
    / "multicity"
    / "global_queries"
    / "figures"
    / "all_queries_ranked_axonometric"
)
OUTPUT_DIR = ROOT / "outputs" / "figure_style_references"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INK = "#263238"
MUTED = "#68747A"
LINE = "#CDD4D7"
PAPER = "#FBFAF7"
BLUE = "#4C78A8"
BLUE_LIGHT = "#E6EEF6"
ORANGE = "#D96C4A"
ORANGE_LIGHT = "#F7E5DE"
GREEN = "#4F8062"
GREEN_LIGHT = "#E5F0E9"


def configure_fonts() -> None:
    candidates = [
        "Microsoft YaHei",
        "Microsoft YaHei UI",
        "Noto Sans CJK SC",
        "SimHei",
        "Arial Unicode MS",
    ]
    installed = {item.name for item in font_manager.fontManager.ttflist}
    for candidate in candidates:
        if candidate in installed:
            plt.rcParams["font.family"] = candidate
            break
    plt.rcParams["axes.unicode_minus"] = False


def clear_axis(ax) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")


def rounded_box(ax, xy, width, height, *, facecolor="white", edgecolor=LINE,
                linewidth=1.2, radius=0.02, zorder=1):
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        zorder=zorder,
    )
    ax.add_patch(box)
    return box


def arrow(ax, start, end, *, color=MUTED, width=1.5, scale=12, zorder=3):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=scale,
        linewidth=width,
        color=color,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def panel_label(ax, letter: str, title: str) -> None:
    ax.text(0.00, 0.99, letter, ha="left", va="top", fontsize=13,
            fontweight="bold", color="white",
            bbox=dict(boxstyle="square,pad=0.25", facecolor=INK, edgecolor=INK))
    ax.text(0.09, 0.985, title, ha="left", va="top", fontsize=13,
            fontweight="bold", color=INK)


def draw_roads(ax, origin=(0.10, 0.16), size=(0.35, 0.25)) -> None:
    x, y = origin
    w, h = size
    roads = [
        [(x, y + 0.06 * h), (x + 0.24 * w, y + 0.55 * h), (x + 0.55 * w, y + 0.45 * h), (x + w, y + 0.92 * h)],
        [(x + 0.08 * w, y + h), (x + 0.34 * w, y + 0.58 * h), (x + 0.62 * w, y + 0.15 * h)],
        [(x + 0.05 * w, y + 0.72 * h), (x + 0.42 * w, y + 0.76 * h), (x + 0.86 * w, y + 0.28 * h)],
        [(x + 0.50 * w, y + h), (x + 0.55 * w, y + 0.45 * h), (x + 0.96 * w, y + 0.08 * h)],
    ]
    for points in roads:
        xs, ys = zip(*points)
        ax.plot(xs, ys, color=MUTED, linewidth=2.0, solid_capstyle="round")
        ax.plot(xs, ys, color="white", linewidth=0.7, solid_capstyle="round")


def draw_buildings(ax, origin=(0.58, 0.16), scale=1.0, roof=True) -> None:
    x0, y0 = origin
    footprints = [
        (0.00, 0.01, 0.09, 0.10),
        (0.12, 0.04, 0.10, 0.07),
        (0.24, 0.00, 0.08, 0.12),
        (0.05, 0.16, 0.12, 0.08),
        (0.21, 0.17, 0.11, 0.10),
    ]
    for x, y, w, h in footprints:
        x = x0 + scale * x
        y = y0 + scale * y
        w *= scale
        h *= scale
        ax.add_patch(Rectangle((x, y), w, h, facecolor="#AEB8BC", edgecolor="white", linewidth=0.6))
        if roof:
            ax.add_patch(
                Polygon(
                    [(x, y + h), (x + w / 2, y + h + 0.035 * scale), (x + w, y + h)],
                    closed=True,
                    facecolor=ORANGE,
                    edgecolor="white",
                    linewidth=0.5,
                )
            )


def draw_house(ax, center=(0.5, 0.5), scale=1.0, pitched=True) -> None:
    cx, cy = center
    w, h = 0.26 * scale, 0.24 * scale
    x, y = cx - w / 2, cy - h / 2
    ax.add_patch(Rectangle((x, y), w, h, facecolor="#AEB8BC", edgecolor=INK, linewidth=1.0))
    if pitched:
        ax.add_patch(
            Polygon(
                [(x, y + h), (cx, y + h + 0.12 * scale), (x + w, y + h)],
                closed=True,
                facecolor=ORANGE,
                edgecolor=INK,
                linewidth=1.0,
            )
        )
        ax.plot([cx, cx], [y + h, y + h + 0.12 * scale], color=INK, linewidth=0.8)
    else:
        ax.add_patch(Rectangle((x, y + h), w, 0.018 * scale, facecolor=BLUE, edgecolor=INK, linewidth=0.8))


def figure_one() -> Path:
    fig = plt.figure(figsize=(18, 8.4), dpi=200, facecolor=PAPER)
    grid = fig.add_gridspec(1, 4, left=0.035, right=0.98, top=0.86, bottom=0.08, wspace=0.10)
    axes = [fig.add_subplot(grid[0, i]) for i in range(4)]
    for ax in axes:
        clear_axis(ax)
    fig.text(0.035, 0.94, "图1  数据基础、同源模型与两条检索路径", fontsize=23,
             fontweight="bold", color=INK, ha="left", va="center")
    fig.text(0.035, 0.895, "样式参考｜从跨城市开放数据到可比较的屋顶信息实验", fontsize=11,
             color=MUTED, ha="left")

    # A. Data construction
    ax = axes[0]
    panel_label(ax, "A", "跨城市案例库")
    rounded_box(ax, (0.02, 0.60), 0.43, 0.25, facecolor=ORANGE_LIGHT, edgecolor=ORANGE)
    draw_buildings(ax, origin=(0.07, 0.64), scale=0.8, roof=True)
    ax.text(0.235, 0.625, "19个城市\nLOD2建筑模型", ha="center", va="top", fontsize=10, color=INK)
    rounded_box(ax, (0.55, 0.60), 0.43, 0.25, facecolor="#F0F2F2")
    draw_roads(ax, origin=(0.60, 0.65), size=(0.30, 0.14))
    ax.text(0.765, 0.625, "OSM道路围合街区", ha="center", va="top", fontsize=10, color=INK)
    arrow(ax, (0.25, 0.57), (0.48, 0.46), color=MUTED)
    arrow(ax, (0.75, 0.57), (0.52, 0.46), color=MUTED)
    rounded_box(ax, (0.16, 0.34), 0.68, 0.12, facecolor="white")
    ax.text(0.50, 0.40, "建筑—街区空间匹配", ha="center", va="center", fontsize=11, fontweight="bold", color=INK)
    arrow(ax, (0.50, 0.33), (0.50, 0.26), color=MUTED)
    ax.text(0.50, 0.235, "79,550个原始候选", ha="center", va="center", fontsize=10, color=MUTED)
    arrow(ax, (0.50, 0.20), (0.50, 0.14), color=MUTED)
    rounded_box(ax, (0.20, 0.02), 0.60, 0.11, facecolor=GREEN_LIGHT, edgecolor=GREEN)
    ax.text(0.50, 0.075, "47,323个有效街区", ha="center", va="center", fontsize=12, fontweight="bold", color=GREEN)

    # B. Same-source models
    ax = axes[1]
    panel_label(ax, "B", "同源模型对照")
    ax.text(0.50, 0.88, "同一建筑对象与街区边界", ha="center", va="center", fontsize=10, color=MUTED)
    rounded_box(ax, (0.02, 0.46), 0.43, 0.34, facecolor=BLUE_LIGHT, edgecolor=BLUE)
    draw_house(ax, center=(0.235, 0.63), scale=0.72, pitched=False)
    ax.text(0.235, 0.49, "单一高度体块", ha="center", va="center", fontsize=11, fontweight="bold", color=BLUE)
    rounded_box(ax, (0.55, 0.46), 0.43, 0.34, facecolor=ORANGE_LIGHT, edgecolor=ORANGE)
    draw_house(ax, center=(0.765, 0.61), scale=0.72, pitched=True)
    ax.text(0.765, 0.49, "保留屋顶几何", ha="center", va="center", fontsize=11, fontweight="bold", color=ORANGE)
    arrow(ax, (0.45, 0.63), (0.55, 0.63), color=INK, width=1.2)
    ax.text(0.50, 0.67, "增加细节", ha="center", va="bottom", fontsize=9, color=MUTED)
    rounded_box(ax, (0.08, 0.24), 0.84, 0.12, facecolor="white")
    ax.text(0.50, 0.30, "高度＝屋顶投影面积加权平均高程－地面高程", ha="center", va="center", fontsize=9.2, color=INK)
    ax.text(0.50, 0.13, "控制数据来源、建筑对象和边界差异\n使两组仅在屋顶细节上不同", ha="center", va="center", fontsize=10, color=MUTED, linespacing=1.5)

    # C. Feature groups
    ax = axes[2]
    panel_label(ax, "C", "形态指标")
    rounded_box(ax, (0.03, 0.47), 0.94, 0.35, facecolor=BLUE_LIGHT, edgecolor=BLUE)
    ax.text(0.08, 0.76, "总体形态｜11项", ha="left", va="center", fontsize=12, fontweight="bold", color=BLUE)
    general_groups = ["街区尺度与形状", "建筑覆盖与高度", "建筑粒度", "布局与方向"]
    for i, label in enumerate(general_groups):
        y = 0.68 - i * 0.065
        ax.add_patch(Rectangle((0.09, y - 0.012), 0.018, 0.018, facecolor=BLUE, edgecolor="none"))
        ax.text(0.13, y, label, ha="left", va="center", fontsize=10, color=INK)
    rounded_box(ax, (0.03, 0.12), 0.94, 0.27, facecolor=ORANGE_LIGHT, edgecolor=ORANGE)
    ax.text(0.08, 0.34, "屋顶特征｜3项", ha="left", va="center", fontsize=12, fontweight="bold", color=ORANGE)
    roof_groups = ["面积加权平均坡度", "屋顶相对落差中位数", "建筑间坡度离散度"]
    for i, label in enumerate(roof_groups):
        y = 0.27 - i * 0.065
        ax.add_patch(Rectangle((0.09, y - 0.012), 0.018, 0.018, facecolor=ORANGE, edgecolor="none"))
        ax.text(0.13, y, label, ha="left", va="center", fontsize=10, color=INK)
    ax.text(0.50, 0.055, "中位数／四分位距稳健标准化", ha="center", va="center", fontsize=9.5, color=MUTED)

    # D. Retrieval paths
    ax = axes[3]
    panel_label(ax, "D", "两条检索路径")
    rounded_box(ax, (0.34, 0.82), 0.32, 0.10, facecolor="white", edgecolor=INK)
    ax.text(0.50, 0.87, "查询街区", ha="center", va="center", fontsize=11, fontweight="bold", color=INK)
    arrow(ax, (0.43, 0.81), (0.25, 0.72), color=BLUE)
    arrow(ax, (0.57, 0.81), (0.75, 0.72), color=ORANGE)
    rounded_box(ax, (0.02, 0.55), 0.45, 0.16, facecolor=BLUE_LIGHT, edgecolor=BLUE)
    ax.text(0.245, 0.655, "总体形态路径", ha="center", va="center", fontsize=11, fontweight="bold", color=BLUE)
    ax.text(0.245, 0.595, "11项指标直接排序", ha="center", va="center", fontsize=9.5, color=INK)
    rounded_box(ax, (0.53, 0.55), 0.45, 0.16, facecolor=ORANGE_LIGHT, edgecolor=ORANGE)
    ax.text(0.755, 0.655, "屋顶增强路径", ha="center", va="center", fontsize=11, fontweight="bold", color=ORANGE)
    ax.text(0.755, 0.595, "先取总体形态前50名", ha="center", va="center", fontsize=9.5, color=INK)
    arrow(ax, (0.245, 0.54), (0.245, 0.43), color=BLUE)
    arrow(ax, (0.755, 0.54), (0.755, 0.43), color=ORANGE)
    rounded_box(ax, (0.08, 0.30), 0.33, 0.12, facecolor="white", edgecolor=BLUE)
    ax.text(0.245, 0.36, "总体距离前5名", ha="center", va="center", fontsize=10, color=BLUE, fontweight="bold")
    rounded_box(ax, (0.58, 0.30), 0.34, 0.12, facecolor="white", edgecolor=ORANGE)
    ax.text(0.75, 0.375, "65% 总体形态", ha="center", va="center", fontsize=9.5, color=INK)
    ax.text(0.75, 0.335, "+ 35% 屋顶特征", ha="center", va="center", fontsize=9.5, color=ORANGE)
    arrow(ax, (0.75, 0.29), (0.75, 0.20), color=ORANGE)
    rounded_box(ax, (0.58, 0.07), 0.34, 0.12, facecolor="white", edgecolor=ORANGE)
    ax.text(0.75, 0.13, "综合排序前5名", ha="center", va="center", fontsize=10, color=ORANGE, fontweight="bold")
    ax.text(0.25, 0.13, "M1", ha="center", va="center", fontsize=18, color=BLUE, fontweight="bold")
    ax.text(0.50, 0.13, "vs.", ha="center", va="center", fontsize=10, color=MUTED)
    ax.text(0.75, 0.025, "M2", ha="center", va="center", fontsize=18, color=ORANGE, fontweight="bold")

    output = OUTPUT_DIR / "figure1_method_reference.png"
    fig.savefig(output, dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)
    return output


def crop_case_figure(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    image = Image.open(path).convert("RGB")
    width, height = image.size
    query = image.crop((0, int(height * 0.18), int(width * 0.23), int(height * 0.88)))
    m1 = image.crop((int(width * 0.22), int(height * 0.05), width, int(height * 0.49)))
    m2 = image.crop((int(width * 0.22), int(height * 0.51), width, int(height * 0.96)))
    return np.asarray(query), np.asarray(m1), np.asarray(m2)


def add_case_panel(fig, spec, title: str, subtitle: str, image_path: Path) -> None:
    panel = fig.add_subplot(spec)
    clear_axis(panel)
    panel.text(0.00, 1.00, title, ha="left", va="top", fontsize=12.5,
               fontweight="bold", color=INK)
    panel.text(0.00, 0.94, subtitle, ha="left", va="top", fontsize=9.5, color=MUTED)
    query, m1, m2 = crop_case_figure(image_path)
    q_ax = panel.inset_axes([0.00, 0.06, 0.23, 0.80])
    q_ax.imshow(query)
    q_ax.axis("off")
    q_ax.text(0.5, 1.02, "查询", transform=q_ax.transAxes, ha="center", va="bottom",
              fontsize=9.5, fontweight="bold", color=INK)
    m1_ax = panel.inset_axes([0.25, 0.48, 0.75, 0.38])
    m1_ax.imshow(m1)
    m1_ax.axis("off")
    m1_ax.text(-0.01, 0.5, "M1\n总体形态", transform=m1_ax.transAxes, ha="right", va="center",
               fontsize=9.2, color=BLUE, fontweight="bold")
    m2_ax = panel.inset_axes([0.25, 0.06, 0.75, 0.39])
    m2_ax.imshow(m2)
    m2_ax.axis("off")
    m2_ax.text(-0.01, 0.5, "M2\n屋顶增强", transform=m2_ax.transAxes, ha="right", va="center",
               fontsize=9.2, color=ORANGE, fontweight="bold")
    panel.plot([0, 1], [0.01, 0.01], color=LINE, linewidth=0.8, transform=panel.transAxes)


def figure_two() -> Path:
    fig = plt.figure(figsize=(18, 11.5), dpi=200, facecolor=PAPER)
    outer = fig.add_gridspec(
        2,
        1,
        height_ratios=[0.38, 0.62],
        left=0.04,
        right=0.98,
        top=0.90,
        bottom=0.05,
        hspace=0.16,
    )
    fig.text(0.04, 0.955, "图2  总体增益、形态代价与代表性案例", fontsize=23,
             fontweight="bold", color=INK, ha="left", va="center")
    fig.text(0.04, 0.918, "样式参考｜简单数值变化＋三组可视化对照", fontsize=11,
             color=MUTED, ha="left")

    summary = outer[0].subgridspec(1, 2, width_ratios=[0.48, 0.52], wspace=0.10)
    overall = fig.add_subplot(summary[0, 0])
    scenarios = fig.add_subplot(summary[0, 1])
    for ax in (overall, scenarios):
        clear_axis(ax)

    panel_label(overall, "A", "50个查询的总体变化")
    overall.text(0.04, 0.80, "屋顶特征距离", fontsize=11, color=INK, va="center")
    overall.text(0.31, 0.80, "0.521", fontsize=19, color=MUTED, fontweight="bold", va="center")
    arrow(overall, (0.43, 0.80), (0.61, 0.80), color=ORANGE, width=2.3, scale=16)
    overall.text(0.66, 0.80, "0.176", fontsize=19, color=ORANGE, fontweight="bold", va="center")
    overall.text(0.84, 0.80, "下降66.3%", fontsize=11, color=ORANGE, fontweight="bold", va="center")
    overall.text(0.04, 0.59, "总体形态距离", fontsize=11, color=INK, va="center")
    overall.text(0.31, 0.59, "0.287", fontsize=19, color=BLUE, fontweight="bold", va="center")
    arrow(overall, (0.43, 0.59), (0.61, 0.59), color=BLUE, width=2.3, scale=16)
    overall.text(0.66, 0.59, "0.334", fontsize=19, color=MUTED, fontweight="bold", va="center")
    overall.text(0.84, 0.59, "上升16.4%", fontsize=11, color=MUTED, fontweight="bold", va="center")
    rounded_box(overall, (0.04, 0.15), 0.39, 0.23, facecolor=ORANGE_LIGHT, edgecolor=ORANGE)
    overall.text(0.235, 0.285, "49 / 50", fontsize=21, fontweight="bold", color=ORANGE, ha="center")
    overall.text(0.235, 0.205, "查询的屋顶距离下降", fontsize=9.5, color=INK, ha="center")
    rounded_box(overall, (0.53, 0.15), 0.39, 0.23, facecolor=BLUE_LIGHT, edgecolor=BLUE)
    overall.text(0.725, 0.285, "3.24 / 5", fontsize=21, fontweight="bold", color=BLUE, ha="center")
    overall.text(0.725, 0.205, "前5名平均被替换", fontsize=9.5, color=INK, ha="center")

    panel_label(scenarios, "B", "五种查询情景的平均屋顶距离降幅")
    labels = ["平缓屋顶*", "中值状态", "大坡度", "高落差", "高离散度"]
    values = [90.9, 55.5, 65.6, 48.4, 57.5]
    ys = np.linspace(0.78, 0.25, len(labels))
    for label, value, y in zip(labels, values, ys):
        scenarios.text(0.04, y, label, ha="left", va="center", fontsize=10.5, color=INK)
        scenarios.add_patch(Rectangle((0.24, y - 0.021), 0.58, 0.042, facecolor="#E7E9E9", edgecolor="none"))
        scenarios.add_patch(Rectangle((0.24, y - 0.021), 0.58 * value / 100, 0.042, facecolor=ORANGE, edgecolor="none"))
        scenarios.text(0.85, y, f"{value:.1f}%", ha="left", va="center", fontsize=10.5,
                       color=ORANGE, fontweight="bold")
    scenarios.text(0.04, 0.10, "* 三项指标零值集中，可能同时反映建模方式，解释需谨慎", ha="left",
                   va="center", fontsize=8.8, color=MUTED)

    cases = outer[1].subgridspec(1, 3, wspace=0.055)
    case_specs = [
        (
            "C1  中值状态｜CZ_PR_000849",
            "屋顶距离 −56.6%｜总体距离 +8.9%｜替换3/5",
            QUERY_FIGURE_DIR / "15_typical_CZ_PR_000849_axonometric.png",
        ),
        (
            "C2  大坡度｜DE_BE_008010",
            "屋顶距离 −75.9%｜总体距离 +22.6%｜替换4/5",
            QUERY_FIGURE_DIR / "16_steep_DE_BE_008010_axonometric.png",
        ),
        (
            "C3  高离散度｜DE_BE_002692",
            "屋顶距离 −58.5%｜总体距离 +13.6%｜替换4/5",
            QUERY_FIGURE_DIR / "13_heterogeneous_DE_BE_002692_axonometric.png",
        ),
    ]
    for index, (title, subtitle, path) in enumerate(case_specs):
        add_case_panel(fig, cases[0, index], title, subtitle, path)

    output = OUTPUT_DIR / "figure2_results_reference.png"
    fig.savefig(output, dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)
    return output


if __name__ == "__main__":
    configure_fonts()
    for generated in (figure_one(), figure_two()):
        print(generated)
