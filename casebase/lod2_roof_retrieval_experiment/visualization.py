"""Publication-oriented figures for the Luxembourg roof retrieval experiment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
import pyvista as pv
from shapely import wkb
from shapely.geometry import MultiPolygon, Polygon

from database import SOURCE_BLOCK_TABLE, SOURCE_BUILDING_TABLE, SOURCE_SURFACE_TABLE


@dataclass(frozen=True)
class SurfaceGeometry:
    surface_type: str
    polygon: Polygon


@dataclass(frozen=True)
class BlockScene:
    block_id: str
    block_geometry: Polygon | MultiPolygon
    footprints: tuple[Polygon, ...]
    surfaces: tuple[SurfaceGeometry, ...]


@dataclass(frozen=True)
class PanelSpec:
    row: int
    column: int
    method: str
    rank: int
    block_id: str
    title: str


def build_comparison_plan(
    query_id: str, retrieval: pd.DataFrame, top_k: int = 5
) -> list[PanelSpec]:
    """Arrange one large query and two rows of ranked candidates."""

    if top_k <= 0:
        raise ValueError("top_k must be positive")
    query_id = str(query_id)
    subset = retrieval[retrieval["query_id"].astype(str) == query_id].copy()
    panels = [
        PanelSpec(
            row=-1,
            column=0,
            method="Query",
            rank=0,
            block_id=query_id,
            title=f"Query\n{query_id}",
        )
    ]
    expected_ranks = list(range(1, top_k + 1))
    for row_index, method in enumerate(("M1", "M2")):
        method_rows = subset[subset["method"] == method].copy()
        method_rows["rank"] = method_rows["rank"].astype(int)
        method_rows = method_rows.sort_values("rank", kind="mergesort")
        if method_rows["rank"].tolist() != expected_ranks:
            raise ValueError(
                f"{query_id} {method} must contain unique ranks {expected_ranks}"
            )
        for candidate in method_rows.itertuples(index=False):
            panels.append(
                PanelSpec(
                    row=row_index,
                    column=int(candidate.rank),
                    method=method,
                    rank=int(candidate.rank),
                    block_id=str(candidate.candidate_id),
                    title=(
                        f"{method} rank {int(candidate.rank)}\n"
                        f"{candidate.candidate_id}\n"
                        f"G {float(candidate.d_general):.2f}  ·  "
                        f"R {float(candidate.d_roof):.2f}"
                    ),
                )
            )
    return panels


def build_overview_deltas(query_summary: pd.DataFrame) -> pd.DataFrame:
    """Create one row per query containing literal M2-minus-M1 changes."""

    metric_map = {
        "slope_wasserstein_deg": "slope_delta",
        "relief_wasserstein": "relief_delta",
        "d_general": "general_delta",
    }
    required = {"query_id", "method", *metric_map}
    missing = sorted(required - set(query_summary.columns))
    if missing:
        raise ValueError(f"Missing overview columns: {missing}")
    methods = query_summary[query_summary["method"].isin(["M1", "M2"])].copy()
    if methods.empty or methods.groupby("query_id")["method"].nunique().min() != 2:
        raise ValueError("Every query must contain both M1 and M2 summaries")
    metadata_columns = [
        column
        for column in ("query_id", "query_role", "stratum")
        if column in methods.columns
    ]
    result = (
        methods[metadata_columns]
        .drop_duplicates("query_id")
        .sort_values("query_id", kind="mergesort")
        .reset_index(drop=True)
    )
    for source, target in metric_map.items():
        pivot = methods.pivot(index="query_id", columns="method", values=source)
        deltas = (pivot["M2"] - pivot["M1"]).round(12)
        result[target] = result["query_id"].map(deltas)
    return result


def _load_geometry(value):
    return wkb.loads(bytes(value))


def fetch_luxembourg_scenes(
    connection, block_ids: Iterable[str], metric_srid: int
) -> dict[str, BlockScene]:
    """Read boundaries, footprints and LOD2 surfaces for requested blocks."""

    selected = sorted({str(block_id) for block_id in block_ids})
    if not selected:
        return {}
    with connection.cursor() as cursor:
        cursor.execute(
            f"""
            SELECT bl.block_id, ST_AsEWKB(ST_Transform(bl.geom, %s))
            FROM {SOURCE_BLOCK_TABLE} AS bl
            WHERE bl.block_id = ANY(%s)
            ORDER BY bl.block_id
            """,
            (int(metric_srid), selected),
        )
        block_rows = cursor.fetchall()
    with connection.cursor() as cursor:
        cursor.execute(
            f"""
            SELECT b.block_id, b.building_id,
                   ST_AsEWKB(ST_Transform(b.geom_2d, %s))
            FROM {SOURCE_BUILDING_TABLE} AS b
            WHERE b.block_id = ANY(%s)
              AND b.geom_2d IS NOT NULL
            ORDER BY b.block_id, b.building_id
            """,
            (int(metric_srid), selected),
        )
        footprint_rows = cursor.fetchall()
    with connection.cursor() as cursor:
        cursor.execute(
            f"""
            SELECT b.block_id, s.surface_type,
                   ST_AsEWKB(ST_Transform(s.geom_3d, %s))
            FROM {SOURCE_BUILDING_TABLE} AS b
            JOIN {SOURCE_SURFACE_TABLE} AS s ON s.building_id = b.building_id
            WHERE b.block_id = ANY(%s)
              AND s.surface_type IN ('RoofSurface', 'WallSurface', 'GroundSurface')
            ORDER BY b.block_id, b.building_id, s.surface_type, s.surface_id
            """,
            (int(metric_srid), selected),
        )
        surface_rows = cursor.fetchall()

    boundaries = {str(block_id): _load_geometry(value) for block_id, value in block_rows}
    footprints_by_building: dict[tuple[str, str], Polygon] = {}
    surfaces: dict[str, list[SurfaceGeometry]] = {block_id: [] for block_id in selected}
    for block_id, building_id, footprint_value in footprint_rows:
        block_id = str(block_id)
        footprint = _load_geometry(footprint_value)
        if isinstance(footprint, Polygon):
            footprints_by_building[(block_id, str(building_id))] = footprint
    for block_id, surface_type, surface_value in surface_rows:
        block_id = str(block_id)
        surface = _load_geometry(surface_value)
        if isinstance(surface, Polygon):
            surfaces[block_id].append(SurfaceGeometry(str(surface_type), surface))
    missing = sorted(set(selected) - set(boundaries))
    if missing:
        raise ValueError("Missing block geometry for: " + ", ".join(missing))
    return {
        block_id: BlockScene(
            block_id=block_id,
            block_geometry=boundaries[block_id],
            footprints=tuple(
                footprint
                for (footprint_block_id, _), footprint in footprints_by_building.items()
                if footprint_block_id == block_id
            ),
            surfaces=tuple(surfaces[block_id]),
        )
        for block_id in selected
    }


def _polygons(geometry: Polygon | MultiPolygon) -> list[Polygon]:
    if isinstance(geometry, Polygon):
        return [geometry]
    if isinstance(geometry, MultiPolygon):
        return list(geometry.geoms)
    raise ValueError(f"Expected Polygon or MultiPolygon, got {geometry.geom_type}")


def _scene_statistics(scene: BlockScene) -> tuple[float, int, float]:
    area = float(scene.block_geometry.area)
    footprint_area = float(sum(footprint.area for footprint in scene.footprints))
    bcr = footprint_area / area if area > 0 else float("nan")
    return area, len(scene.footprints), bcr


def _panel_annotation(scene: BlockScene) -> str:
    area, building_count, bcr = _scene_statistics(scene)
    return f"{area:,.0f} m²  ·  {building_count} bldg  ·  BCR {bcr:.2f}"


def _style_axis(axis, method: str) -> None:
    colors = {"Query": "#343434", "M1": "#4C78A8", "M2": "#D66B4F"}
    for spine in axis.spines.values():
        spine.set_color(colors[method])
        spine.set_linewidth(1.8 if method == "Query" else 0.9)
    axis.set_facecolor("#FCFCFB" if method == "Query" else "white")


def _draw_plan_scene(axis, scene: BlockScene, method: str) -> None:
    for block_polygon in _polygons(scene.block_geometry):
        x, y = block_polygon.exterior.xy
        axis.fill(x, y, color="#E5D9D2", alpha=0.55, zorder=1)
        axis.plot(x, y, color="#2F2F2F", linewidth=1.25, zorder=3)
        for interior in block_polygon.interiors:
            hole_x, hole_y = interior.xy
            axis.fill(hole_x, hole_y, color="white", zorder=2)
            axis.plot(hole_x, hole_y, color="#2F2F2F", linewidth=0.7, zorder=3)
    for footprint in scene.footprints:
        x, y = footprint.exterior.xy
        axis.fill(
            x,
            y,
            color="#6687B5",
            alpha=0.78,
            edgecolor="#253C5A",
            linewidth=0.45,
            zorder=4,
        )
    min_x, min_y, max_x, max_y = scene.block_geometry.bounds
    span = max(max_x - min_x, max_y - min_y, 1.0)
    center_x = (min_x + max_x) / 2.0
    center_y = (min_y + max_y) / 2.0
    half_range = span * 0.58
    axis.set_xlim(center_x - half_range, center_x + half_range)
    axis.set_ylim(center_y - half_range, center_y + half_range)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xticks([])
    axis.set_yticks([])
    axis.text(
        0.5,
        0.015,
        _panel_annotation(scene),
        transform=axis.transAxes,
        ha="center",
        va="bottom",
        fontsize=7.2,
        color="#555555",
        bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none", "pad": 1.5},
        zorder=10,
    )
    _style_axis(axis, method)


def _build_comparison_axes(fig, top_k: int):
    grid = GridSpec(
        2,
        top_k + 1,
        figure=fig,
        width_ratios=[1.55, *([1.0] * top_k)],
        wspace=0.10,
        hspace=0.22,
    )
    query_axis = fig.add_subplot(grid[:, 0])
    candidate_axes = {
        (row, column): fig.add_subplot(grid[row, column])
        for row in range(2)
        for column in range(1, top_k + 1)
    }
    return query_axis, candidate_axes


def _apply_panel_titles(
    query_axis, candidate_axes, plan: list[PanelSpec], scenes: dict[str, BlockScene], draw
) -> None:
    query_panel = plan[0]
    draw(query_axis, scenes[query_panel.block_id], query_panel.method)
    query_axis.set_title(query_panel.title, fontsize=11, fontweight="bold", pad=8)
    for panel in plan[1:]:
        axis = candidate_axes[(panel.row, panel.column)]
        draw(axis, scenes[panel.block_id], panel.method)
        axis.set_title(
            panel.title,
            fontsize=8.2,
            color="#4C78A8" if panel.method == "M1" else "#D66B4F",
            pad=5,
        )


def _check_scene_coverage(plan: list[PanelSpec], scenes: dict[str, BlockScene]) -> None:
    missing = sorted({panel.block_id for panel in plan} - set(scenes))
    if missing:
        raise ValueError("Missing visualization scenes for: " + ", ".join(missing))


def save_query_plan_comparison(
    query_id: str,
    retrieval: pd.DataFrame,
    scenes: dict[str, BlockScene],
    output_path: str | Path,
    top_k: int = 5,
) -> Path:
    """Save a footprint-focused plan comparison with per-block normalization."""

    plan = build_comparison_plan(query_id, retrieval, top_k=top_k)
    _check_scene_coverage(plan, scenes)
    fig = plt.figure(figsize=(18, 9.5), facecolor="white")
    query_axis, candidate_axes = _build_comparison_axes(fig, top_k)
    _apply_panel_titles(query_axis, candidate_axes, plan, scenes, _draw_plan_scene)
    fig.text(0.185, 0.735, "M1 baseline", rotation=90, color="#4C78A8", fontsize=11)
    fig.text(0.185, 0.285, "M2 roof-aware", rotation=90, color="#D66B4F", fontsize=11)
    fig.suptitle(
        f"Query {query_id} — plan comparison (each block fitted independently)",
        fontsize=15,
        y=0.975,
    )
    fig.subplots_adjust(left=0.025, right=0.99, top=0.90, bottom=0.04)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=230, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def _surface_z_values(scene: BlockScene) -> np.ndarray:
    values = [
        float(coordinate[2])
        for surface in scene.surfaces
        for coordinate in surface.polygon.exterior.coords
        if len(coordinate) >= 3 and np.isfinite(coordinate[2])
    ]
    return np.asarray(values, dtype=float)


def _surface_mesh(
    polygon: Polygon, center_x: float, center_y: float, base_z: float
) -> pv.PolyData | None:
    coordinates = list(polygon.exterior.coords)[:-1]
    if len(coordinates) < 3 or any(len(coordinate) < 3 for coordinate in coordinates):
        return None
    points = np.asarray(
        [
            [
                float(coordinate[0]) - center_x,
                float(coordinate[1]) - center_y,
                float(coordinate[2]) - base_z,
            ]
            for coordinate in coordinates
        ],
        dtype=float,
    )
    faces = np.asarray([len(points), *range(len(points))], dtype=int)
    mesh = pv.PolyData(points, faces)
    return mesh.triangulate() if mesh.n_cells else None


def _flat_mesh(
    polygon: Polygon, center_x: float, center_y: float, z_value: float
) -> pv.PolyData | None:
    coordinates = list(polygon.exterior.coords)[:-1]
    if len(coordinates) < 3:
        return None
    points = np.asarray(
        [[x - center_x, y - center_y, z_value] for x, y, *_ in coordinates],
        dtype=float,
    )
    faces = np.asarray([len(points), *range(len(points))], dtype=int)
    mesh = pv.PolyData(points, faces)
    return mesh.triangulate() if mesh.n_cells else None


def _configure_axonometric_camera(plotter) -> None:
    """Use an isometric direction with orthographic, not perspective, projection."""

    plotter.camera_position = "iso"
    plotter.enable_parallel_projection()
    plotter.reset_camera()
    plotter.camera.zoom(1.35)


def _render_block_axonometric(
    scene: BlockScene, window_size: tuple[int, int] = (620, 520)
) -> np.ndarray:
    min_x, min_y, max_x, max_y = scene.block_geometry.bounds
    center_x = (min_x + max_x) / 2.0
    center_y = (min_y + max_y) / 2.0
    z_values = _surface_z_values(scene)
    base_z = float(z_values.min()) if z_values.size else 0.0
    plotter = pv.Plotter(off_screen=True, window_size=list(window_size))
    plotter.set_background("#FFFFFF")
    try:
        plotter.enable_anti_aliasing("ssaa")
    except Exception:
        pass

    block_meshes = [
        mesh
        for polygon in _polygons(scene.block_geometry)
        if (mesh := _flat_mesh(polygon, center_x, center_y, -0.25)) is not None
    ]
    for mesh in block_meshes:
        plotter.add_mesh(mesh, color="#E0D0C8", opacity=0.42, show_edges=False)
        edges = mesh.extract_feature_edges(
            boundary_edges=True,
            feature_edges=False,
            manifold_edges=False,
        )
        plotter.add_mesh(edges, color="#242424", line_width=2.2)

    color_by_type = {
        "RoofSurface": "#F1A493",
        "WallSurface": "#F8EFEA",
        "GroundSurface": "#C1A4A0",
    }
    surface_meshes = []
    for surface in scene.surfaces:
        mesh = _surface_mesh(surface.polygon, center_x, center_y, base_z)
        if mesh is None or mesh.n_cells == 0:
            continue
        color = pv.Color(color_by_type.get(surface.surface_type, "#DDD6D2")).int_rgb
        mesh["color"] = np.tile(color, (mesh.n_cells, 1))
        surface_meshes.append(mesh)
    if surface_meshes:
        building_mesh = (
            surface_meshes[0].merge(surface_meshes[1:])
            if len(surface_meshes) > 1
            else surface_meshes[0]
        )
        plotter.add_mesh(
            building_mesh,
            scalars="color",
            rgb=True,
            show_edges=False,
            ambient=0.32,
            diffuse=0.68,
            specular=0.08,
        )
        edges = building_mesh.extract_feature_edges(
            boundary_edges=True,
            feature_edges=True,
            feature_angle=30,
            manifold_edges=False,
        )
        plotter.add_mesh(edges, color="#383838", line_width=0.8)

    _configure_axonometric_camera(plotter)
    image = plotter.screenshot(return_img=True)
    plotter.close()
    return image


def _draw_axonometric_scene(axis, scene: BlockScene, method: str, image: np.ndarray) -> None:
    axis.imshow(image)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.text(
        0.5,
        0.015,
        _panel_annotation(scene),
        transform=axis.transAxes,
        ha="center",
        va="bottom",
        fontsize=7.2,
        color="#555555",
        bbox={"facecolor": "white", "alpha": 0.84, "edgecolor": "none", "pad": 1.5},
    )
    _style_axis(axis, method)


def save_query_axonometric_comparison(
    query_id: str,
    retrieval: pd.DataFrame,
    scenes: dict[str, BlockScene],
    output_path: str | Path,
    top_k: int = 5,
) -> Path:
    """Save a clean LOD2 axonometric comparison fitted per block."""

    plan = build_comparison_plan(query_id, retrieval, top_k=top_k)
    _check_scene_coverage(plan, scenes)
    image_cache = {
        block_id: _render_block_axonometric(scenes[block_id])
        for block_id in {panel.block_id for panel in plan}
    }
    fig = plt.figure(figsize=(18, 9.5), facecolor="white")
    query_axis, candidate_axes = _build_comparison_axes(fig, top_k)
    query_panel = plan[0]
    _draw_axonometric_scene(
        query_axis,
        scenes[query_panel.block_id],
        query_panel.method,
        image_cache[query_panel.block_id],
    )
    query_axis.set_title(query_panel.title, fontsize=11, fontweight="bold", pad=8)
    for panel in plan[1:]:
        axis = candidate_axes[(panel.row, panel.column)]
        _draw_axonometric_scene(
            axis,
            scenes[panel.block_id],
            panel.method,
            image_cache[panel.block_id],
        )
        axis.set_title(
            panel.title,
            fontsize=8.2,
            color="#4C78A8" if panel.method == "M1" else "#D66B4F",
            pad=5,
        )
    fig.text(0.185, 0.735, "M1 baseline", rotation=90, color="#4C78A8", fontsize=11)
    fig.text(0.185, 0.285, "M2 roof-aware", rotation=90, color="#D66B4F", fontsize=11)
    fig.suptitle(
        f"Query {query_id} — LOD2 axonometric comparison (each block fitted independently)",
        fontsize=15,
        y=0.975,
    )
    fig.subplots_adjust(left=0.025, right=0.99, top=0.90, bottom=0.04)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=230, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def make_formal_overview_figure(
    query_summary: pd.DataFrame, output_path: str | Path | None = None
) -> plt.Figure:
    """Plot compact query-level M2-minus-M1 changes in three stacked panels."""

    deltas = build_overview_deltas(query_summary)
    specs = (
        ("slope_delta", "Building-slope distribution", "M2 − M1 (degrees)"),
        ("relief_delta", "Roof-relief distribution", "M2 − M1"),
        ("general_delta", "General form", "M2 − M1"),
    )
    x_positions = np.arange(len(deltas))
    fig, axes = plt.subplots(3, 1, figsize=(14, 9.5), sharex=True)
    for axis, (field, title, ylabel) in zip(axes, specs):
        values = deltas[field].to_numpy(dtype=float)
        axis.axhline(0.0, color="#777777", linewidth=1.0)
        axis.vlines(x_positions, 0.0, values, color="#D1D1D1", linewidth=1.1)
        regular = deltas.get("query_role", pd.Series("regular", index=deltas.index)) != "boundary"
        boundary = ~regular
        colors = np.where(values < 0, "#3E8E7E", "#D66B4F")
        axis.scatter(
            x_positions[regular],
            values[regular],
            c=colors[regular],
            s=36,
            marker="o",
            zorder=3,
        )
        axis.scatter(
            x_positions[boundary],
            values[boundary],
            c=colors[boundary],
            s=76,
            marker="*",
            edgecolor="#333333",
            linewidth=0.35,
            zorder=4,
        )
        improved_count = int(np.sum(values < 0))
        axis.set_title(
            f"{title}  ·  closer in {improved_count}/{len(values)} queries",
            loc="left",
            fontsize=11,
            fontweight="bold",
        )
        axis.set_ylabel(ylabel, fontsize=9)
        axis.grid(axis="y", color="#ECECEC", linewidth=0.8)
        axis.spines[["top", "right"]].set_visible(False)
    labels = [
        ("* " if role == "boundary" else "") + query_id.replace("LU_LU_", "")
        for query_id, role in zip(
            deltas["query_id"].astype(str),
            deltas.get("query_role", pd.Series("regular", index=deltas.index)),
        )
    ]
    axes[-1].set_xticks(x_positions, labels, rotation=55, ha="right", fontsize=8)
    axes[-1].set_xlabel("Formal query ID  (* boundary case)")
    fig.suptitle(
        "Luxembourg formal retrieval: query-level change after adding roof information",
        fontsize=14,
        y=0.99,
    )
    fig.text(
        0.995,
        0.01,
        "Green: M2 is closer · Orange: M2 is farther",
        ha="right",
        va="bottom",
        fontsize=9,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.975))
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=240, bbox_inches="tight", facecolor="white")
    return fig
