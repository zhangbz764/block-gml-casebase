"""Static comparison figures for the Luxembourg formal retrieval experiment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd
import pyvista as pv
from shapely import wkb
from shapely.geometry import MultiPolygon, Polygon

from database import SOURCE_BLOCK_TABLE, SOURCE_BUILDING_TABLE, SOURCE_SURFACE_TABLE
from roof_geometry import measure_polygon_3d


@dataclass(frozen=True)
class SurfaceGeometry:
    surface_type: str
    polygon: Polygon


@dataclass(frozen=True)
class BlockScene:
    block_id: str
    block_geometry: Polygon | MultiPolygon
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
    """Arrange one repeated query and each method's ranked candidates in two rows."""

    if top_k <= 0:
        raise ValueError("top_k must be positive")
    query_id = str(query_id)
    subset = retrieval[retrieval["query_id"].astype(str) == query_id].copy()
    panels: list[PanelSpec] = []
    expected_ranks = list(range(1, top_k + 1))
    for row_index, method in enumerate(("M1", "M2")):
        method_rows = subset[subset["method"] == method].copy()
        method_rows["rank"] = method_rows["rank"].astype(int)
        method_rows = method_rows.sort_values("rank", kind="mergesort")
        if method_rows["rank"].tolist() != expected_ranks:
            raise ValueError(
                f"{query_id} {method} must contain unique ranks {expected_ranks}"
            )
        panels.append(
            PanelSpec(
                row=row_index,
                column=0,
                method="Query",
                rank=0,
                block_id=query_id,
                title=f"Query\n{query_id}",
            )
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
                        f"general={float(candidate.d_general):.2f}  "
                        f"roof={float(candidate.d_roof):.2f}"
                    ),
                )
            )
    return panels


def _load_geometry(value):
    return wkb.loads(bytes(value))


def fetch_luxembourg_scenes(
    connection, block_ids: Iterable[str], metric_srid: int
) -> dict[str, BlockScene]:
    """Read block boundaries and LOD2 surfaces for a requested block set."""

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
    surfaces: dict[str, list[SurfaceGeometry]] = {block_id: [] for block_id in selected}
    for block_id, surface_type, value in surface_rows:
        polygon = _load_geometry(value)
        if isinstance(polygon, Polygon):
            surfaces[str(block_id)].append(SurfaceGeometry(str(surface_type), polygon))
    missing = sorted(set(selected) - set(boundaries))
    if missing:
        raise ValueError("Missing block geometry for: " + ", ".join(missing))
    return {
        block_id: BlockScene(
            block_id=block_id,
            block_geometry=boundaries[block_id],
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


def _surface_z_values(scene: BlockScene) -> np.ndarray:
    values = [
        float(coordinate[2])
        for surface in scene.surfaces
        for coordinate in surface.polygon.exterior.coords
        if len(coordinate) >= 3 and np.isfinite(coordinate[2])
    ]
    return np.asarray(values, dtype=float)


def _scene_scale(scene: BlockScene) -> float:
    min_x, min_y, max_x, max_y = scene.block_geometry.bounds
    z_values = _surface_z_values(scene)
    height = float(np.ptp(z_values)) if z_values.size else 0.0
    return max(float(max_x - min_x), float(max_y - min_y), 1.5 * height, 1.0)


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


def _render_block_thumbnail(
    scene: BlockScene,
    shared_scale: float,
    slope_range: tuple[float, float],
    window_size: tuple[int, int] = (480, 390),
) -> np.ndarray:
    min_x, min_y, max_x, max_y = scene.block_geometry.bounds
    center_x = (min_x + max_x) / 2.0
    center_y = (min_y + max_y) / 2.0
    z_values = _surface_z_values(scene)
    base_z = float(z_values.min()) if z_values.size else 0.0
    plotter = pv.Plotter(off_screen=True, window_size=list(window_size))
    plotter.set_background("#FFFFFF")
    colormap = plt.get_cmap("magma")
    normalization = Normalize(vmin=slope_range[0], vmax=slope_range[1], clip=True)

    for block_polygon in _polygons(scene.block_geometry):
        ring = list(block_polygon.exterior.coords)
        points = np.asarray(
            [[x - center_x, y - center_y, -0.3] for x, y, *_ in ring], dtype=float
        )
        if len(points) >= 2:
            plotter.add_mesh(
                pv.lines_from_points(points, close=True),
                color="#2F2F2F",
                line_width=3,
            )

    for surface in scene.surfaces:
        if surface.surface_type == "GroundSurface":
            continue
        mesh = _surface_mesh(surface.polygon, center_x, center_y, base_z)
        if mesh is None or mesh.n_cells == 0:
            continue
        if surface.surface_type == "RoofSurface":
            try:
                slope = measure_polygon_3d(surface.polygon).mean_slope_deg
                color = colormap(normalization(slope))[:3]
            except ValueError:
                color = "#D8A39D"
            show_edges = True
            edge_color = "#4A3532"
        else:
            color = "#F1E9E4"
            show_edges = False
            edge_color = None
        plotter.add_mesh(
            mesh,
            color=color,
            show_edges=show_edges,
            edge_color=edge_color,
            line_width=0.7,
            smooth_shading=False,
        )

    camera_distance = shared_scale * 1.8
    plotter.camera_position = [
        (camera_distance, -camera_distance, camera_distance * 0.85),
        (0.0, 0.0, shared_scale * 0.12),
        (0.0, 0.0, 1.0),
    ]
    plotter.camera.parallel_projection = True
    plotter.camera.parallel_scale = shared_scale * 0.72
    image = plotter.screenshot(return_img=True)
    plotter.close()
    return image


def make_formal_overview_figure(
    query_summary: pd.DataFrame, output_path: str | Path | None = None
) -> plt.Figure:
    """Draw paired M1/M2 values for three interpretable retrieval outcomes."""

    metric_specs = (
        ("slope_wasserstein_deg", "Building-slope distribution distance (deg)"),
        ("relief_wasserstein", "Roof-relief distribution distance"),
        ("d_general", "General-form distance"),
    )
    required = {"query_id", "method", *(metric for metric, _ in metric_specs)}
    missing = sorted(required - set(query_summary.columns))
    if missing:
        raise ValueError(f"Missing overview columns: {missing}")
    methods = query_summary[query_summary["method"].isin(["M1", "M2"])].copy()
    if methods.groupby("query_id")["method"].nunique().min() != 2:
        raise ValueError("Every query must contain both M1 and M2 summaries")
    metadata_columns = [
        column for column in ("query_id", "query_role", "stratum")
        if column in methods.columns
    ]
    metadata = methods[metadata_columns].drop_duplicates("query_id")
    if "query_role" in metadata.columns:
        metadata = metadata.sort_values(
            ["query_role", "stratum", "query_id"], kind="mergesort"
        )
    else:
        metadata = metadata.sort_values("query_id", kind="mergesort")
    query_ids = metadata["query_id"].astype(str).tolist()
    labels = []
    for row in metadata.to_dict("records"):
        marker = "* " if row.get("query_role") == "boundary" else ""
        labels.append(marker + str(row["query_id"]))

    fig, axes = plt.subplots(1, 3, figsize=(17, 9), sharey=True)
    y_positions = np.arange(len(query_ids))
    for axis, (metric, title) in zip(axes, metric_specs):
        pivot = methods.pivot(index="query_id", columns="method", values=metric).reindex(query_ids)
        for y_position, (_, values) in enumerate(pivot.iterrows()):
            axis.plot(
                [values["M1"], values["M2"]],
                [y_position, y_position],
                color="#C7C7C7",
                linewidth=1.2,
                zorder=1,
            )
        axis.scatter(pivot["M1"], y_positions, color="#4C78A8", s=30, label="M1", zorder=2)
        axis.scatter(pivot["M2"], y_positions, color="#E07A5F", s=30, label="M2", zorder=3)
        axis.set_title(title, fontsize=11)
        axis.grid(axis="x", color="#EAEAEA", linewidth=0.8)
        axis.spines[["top", "right", "left"]].set_visible(False)
    axes[0].set_yticks(y_positions, labels)
    axes[0].invert_yaxis()
    axes[0].legend(loc="lower right", frameon=False)
    fig.suptitle(
        "Luxembourg formal queries: M1 vs M2 (lower is closer; * = boundary case)",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=240, bbox_inches="tight", facecolor="white")
    return fig


def save_query_comparison(
    query_id: str,
    retrieval: pd.DataFrame,
    scenes: dict[str, BlockScene],
    output_path: str | Path,
    top_k: int = 5,
    slope_range: tuple[float, float] = (0.0, 50.0),
) -> Path:
    """Save a 2x6 query/M1/M2 LOD2 comparison sheet."""

    plan = build_comparison_plan(query_id, retrieval, top_k=top_k)
    missing = sorted({panel.block_id for panel in plan} - set(scenes))
    if missing:
        raise ValueError("Missing visualization scenes for: " + ", ".join(missing))
    shared_scale = max(_scene_scale(scenes[panel.block_id]) for panel in plan)
    image_cache = {
        block_id: _render_block_thumbnail(
            scenes[block_id], shared_scale=shared_scale, slope_range=slope_range
        )
        for block_id in {panel.block_id for panel in plan}
    }

    fig, axes = plt.subplots(2, top_k + 1, figsize=(21, 8.3))
    border_colors = {"Query": "#2F2F2F", "M1": "#4C78A8", "M2": "#E07A5F"}
    for panel in plan:
        axis = axes[panel.row, panel.column]
        axis.imshow(image_cache[panel.block_id])
        axis.set_title(panel.title, fontsize=9, color=border_colors[panel.method])
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_color(border_colors[panel.method])
            spine.set_linewidth(2.2 if panel.method == "Query" else 1.3)
    axes[0, 0].set_ylabel("M1 baseline", fontsize=12, labelpad=12)
    axes[1, 0].set_ylabel("M2 roof-aware", fontsize=12, labelpad=12)
    normalizer = Normalize(vmin=slope_range[0], vmax=slope_range[1])
    color_axis = fig.add_axes((0.40, 0.035, 0.22, 0.018))
    colorbar = fig.colorbar(
        ScalarMappable(norm=normalizer, cmap="magma"),
        cax=color_axis,
        orientation="horizontal",
    )
    colorbar.set_label("Roof-surface slope (degrees)", fontsize=9)
    fig.suptitle(f"Query {query_id}: shared scale and isometric view", fontsize=15)
    fig.subplots_adjust(left=0.04, right=0.99, top=0.88, bottom=0.10, wspace=0.04, hspace=0.20)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path
