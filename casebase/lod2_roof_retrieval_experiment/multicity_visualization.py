"""Visual diagnostics for the multicity LOD2 roof-retrieval experiment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Protocol

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
from shapely import wkb
from shapely.geometry import MultiPolygon, Polygon

SceneKey = tuple[str, str]


class CitySourceLike(Protocol):
    city: str
    display_name: str
    metric_srid: int
    block_table: str
    building_table: str
    surface_table: str


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


def _as_base_scene(scene: BlockScene):
    """Convert to the existing renderer's scene type only when drawing begins."""

    from visualization import BlockScene as BaseBlockScene
    from visualization import SurfaceGeometry as BaseSurfaceGeometry

    return BaseBlockScene(
        block_id=scene.block_id,
        block_geometry=scene.block_geometry,
        footprints=scene.footprints,
        surfaces=tuple(
            BaseSurfaceGeometry(surface.surface_type, surface.polygon)
            for surface in scene.surfaces
        ),
    )


@dataclass(frozen=True)
class MulticityPanelSpec:
    row: int
    column: int
    method: str
    rank: int
    city: str
    block_id: str
    title: str

    @property
    def scene_key(self) -> SceneKey:
        return self.city, self.block_id


def _display_city(city: str) -> str:
    return str(city).replace("_", " ").title()


def _require_columns(frame: pd.DataFrame, columns: set[str], label: str) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing columns: {missing}")


def build_multicity_comparison_plan(
    query_id: str,
    retrieval: pd.DataFrame,
    top_k: int = 5,
) -> list[MulticityPanelSpec]:
    """Arrange one query and two city-aware rows of ranked candidates."""

    if top_k <= 0:
        raise ValueError("top_k must be positive")
    required = {
        "query_id",
        "query_city",
        "method",
        "rank",
        "candidate_id",
        "candidate_city",
        "d_general",
        "d_roof",
    }
    _require_columns(retrieval, required, "retrieval")
    query_id = str(query_id)
    subset = retrieval[retrieval["query_id"].astype(str) == query_id].copy()
    if subset.empty:
        raise ValueError(f"No retrieval rows found for query {query_id}")
    query_cities = subset["query_city"].dropna().astype(str).unique().tolist()
    if len(query_cities) != 1:
        raise ValueError(f"{query_id} must have exactly one query_city")
    query_city = query_cities[0]
    category_suffix = ""
    if "query_category_label" in subset.columns:
        category_labels = (
            subset["query_category_label"].dropna().astype(str).unique().tolist()
        )
        if len(category_labels) == 1:
            category_suffix = f" · {category_labels[0]}"
    panels = [
        MulticityPanelSpec(
            row=-1,
            column=0,
            method="Query",
            rank=0,
            city=query_city,
            block_id=query_id,
            title=f"Query · {_display_city(query_city)}{category_suffix}\n{query_id}",
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
                MulticityPanelSpec(
                    row=row_index,
                    column=int(candidate.rank),
                    method=method,
                    rank=int(candidate.rank),
                    city=str(candidate.candidate_city),
                    block_id=str(candidate.candidate_id),
                    title=(
                        f"{method} rank {int(candidate.rank)} · "
                        f"{_display_city(candidate.candidate_city)}\n"
                        f"{candidate.candidate_id}\n"
                        f"G {float(candidate.d_general):.2f}  ·  "
                        f"R {float(candidate.d_roof):.2f}"
                    ),
                )
            )
    return panels


def scene_keys_for_queries(
    query_ids: Iterable[str], retrieval: pd.DataFrame, top_k: int = 5
) -> list[SceneKey]:
    """Return the unique city/block keys needed by several case figures."""

    keys = {
        panel.scene_key
        for query_id in query_ids
        for panel in build_multicity_comparison_plan(query_id, retrieval, top_k)
    }
    return sorted(keys)


def _polygon_parts(value) -> tuple[Polygon, ...]:
    geometry = wkb.loads(bytes(value))
    if isinstance(geometry, Polygon):
        return (geometry,)
    if isinstance(geometry, MultiPolygon):
        return tuple(geometry.geoms)
    raise ValueError(f"Expected Polygon or MultiPolygon, got {geometry.geom_type}")


def _area_geometry(value) -> Polygon | MultiPolygon:
    geometry = wkb.loads(bytes(value))
    if not isinstance(geometry, (Polygon, MultiPolygon)):
        raise ValueError(f"Expected Polygon or MultiPolygon, got {geometry.geom_type}")
    return geometry


def fetch_multicity_scenes(
    connection,
    sources: Mapping[str, CitySourceLike],
    scene_keys: Iterable[SceneKey],
) -> dict[SceneKey, BlockScene]:
    """Read only the block geometries needed for the requested figures."""

    requested: dict[str, list[str]] = {}
    for city, block_id in scene_keys:
        requested.setdefault(str(city), []).append(str(block_id))
    scenes: dict[SceneKey, BlockScene] = {}
    for city in sorted(requested):
        if city not in sources:
            raise ValueError(f"No catalog source found for city {city!r}")
        source = sources[city]
        block_ids = sorted(set(requested[city]))
        with connection.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT bl.block_id, ST_AsEWKB(ST_Transform(bl.geom, %s))
                FROM {source.block_table} AS bl
                WHERE bl.block_id = ANY(%s)
                ORDER BY bl.block_id
                """,
                (source.metric_srid, block_ids),
            )
            block_rows = cursor.fetchall()
        with connection.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT b.block_id, b.building_id,
                       ST_AsEWKB(ST_Transform(b.geom_2d, %s))
                FROM {source.building_table} AS b
                WHERE b.block_id = ANY(%s) AND b.geom_2d IS NOT NULL
                ORDER BY b.block_id, b.building_id
                """,
                (source.metric_srid, block_ids),
            )
            footprint_rows = cursor.fetchall()
        with connection.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT b.block_id, s.surface_type,
                       ST_AsEWKB(ST_Transform(s.geom_3d, %s))
                FROM {source.building_table} AS b
                JOIN {source.surface_table} AS s ON s.building_id = b.building_id
                WHERE b.block_id = ANY(%s)
                  AND s.surface_type IN ('RoofSurface', 'WallSurface', 'GroundSurface')
                  AND s.geom_3d IS NOT NULL
                ORDER BY b.block_id, b.building_id, s.surface_type, s.surface_id
                """,
                (source.metric_srid, block_ids),
            )
            surface_rows = cursor.fetchall()

        footprints: dict[str, list[Polygon]] = {block_id: [] for block_id in block_ids}
        for block_id, _, geometry_wkb in footprint_rows:
            footprints.setdefault(str(block_id), []).extend(_polygon_parts(geometry_wkb))
        surfaces: dict[str, list[SurfaceGeometry]] = {
            block_id: [] for block_id in block_ids
        }
        for block_id, surface_type, geometry_wkb in surface_rows:
            surfaces.setdefault(str(block_id), []).extend(
                SurfaceGeometry(str(surface_type), polygon)
                for polygon in _polygon_parts(geometry_wkb)
            )
        for block_id, geometry_wkb in block_rows:
            block_id = str(block_id)
            scenes[(city, block_id)] = BlockScene(
                block_id=block_id,
                block_geometry=_area_geometry(geometry_wkb),
                footprints=tuple(footprints.get(block_id, [])),
                surfaces=tuple(surfaces.get(block_id, [])),
            )
    return scenes


def _build_comparison_axes(fig, top_k: int):
    grid = GridSpec(
        2,
        top_k + 1,
        figure=fig,
        width_ratios=[1.58, *([1.0] * top_k)],
        wspace=0.10,
        hspace=0.24,
    )
    query_axis = fig.add_subplot(grid[:, 0])
    candidate_axes = {
        (row, column): fig.add_subplot(grid[row, column])
        for row in range(2)
        for column in range(1, top_k + 1)
    }
    return query_axis, candidate_axes


def _check_scene_coverage(
    plan: list[MulticityPanelSpec], scenes: Mapping[SceneKey, BlockScene]
) -> None:
    missing = sorted({panel.scene_key for panel in plan} - set(scenes))
    if missing:
        readable = ", ".join(f"{city}/{block_id}" for city, block_id in missing)
        raise ValueError("Missing visualization scenes for: " + readable)


def _add_titles_and_labels(fig, query_id: str) -> None:
    fig.text(0.188, 0.735, "M1 baseline", rotation=90, color="#4C78A8", fontsize=11)
    fig.text(0.188, 0.285, "M2 roof-aware", rotation=90, color="#D66B4F", fontsize=11)
    fig.suptitle(
        f"Query {query_id} — multicity top-5 comparison",
        fontsize=15,
        y=0.976,
    )
    fig.subplots_adjust(left=0.025, right=0.99, top=0.89, bottom=0.04)


def save_multicity_plan_comparison(
    query_id: str,
    retrieval: pd.DataFrame,
    scenes: Mapping[SceneKey, BlockScene],
    output_path: str | Path,
    top_k: int = 5,
) -> Path:
    """Save one query with M1 and M2 candidates as normalized plan views."""

    from visualization import _draw_plan_scene

    plan = build_multicity_comparison_plan(query_id, retrieval, top_k)
    _check_scene_coverage(plan, scenes)
    fig = plt.figure(figsize=(18, 9.5), facecolor="white")
    query_axis, candidate_axes = _build_comparison_axes(fig, top_k)
    query = plan[0]
    _draw_plan_scene(query_axis, _as_base_scene(scenes[query.scene_key]), query.method)
    query_axis.set_title(query.title, fontsize=11, fontweight="bold", pad=8)
    for panel in plan[1:]:
        axis = candidate_axes[(panel.row, panel.column)]
        _draw_plan_scene(axis, _as_base_scene(scenes[panel.scene_key]), panel.method)
        axis.set_title(
            panel.title,
            fontsize=8.0,
            color="#4C78A8" if panel.method == "M1" else "#D66B4F",
            pad=5,
        )
    _add_titles_and_labels(fig, query_id)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=230, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def save_multicity_axonometric_comparison(
    query_id: str,
    retrieval: pd.DataFrame,
    scenes: Mapping[SceneKey, BlockScene],
    output_path: str | Path,
    top_k: int = 5,
) -> Path:
    """Save the same comparison using each block's detailed LOD2 geometry."""

    from visualization import _draw_axonometric_scene, _render_block_axonometric

    plan = build_multicity_comparison_plan(query_id, retrieval, top_k)
    _check_scene_coverage(plan, scenes)
    image_cache = {
        panel.scene_key: _render_block_axonometric(_as_base_scene(scenes[panel.scene_key]))
        for panel in plan
    }
    fig = plt.figure(figsize=(18, 9.5), facecolor="white")
    query_axis, candidate_axes = _build_comparison_axes(fig, top_k)
    query = plan[0]
    _draw_axonometric_scene(
        query_axis,
        _as_base_scene(scenes[query.scene_key]),
        query.method,
        image_cache[query.scene_key],
    )
    query_axis.set_title(query.title, fontsize=11, fontweight="bold", pad=8)
    for panel in plan[1:]:
        axis = candidate_axes[(panel.row, panel.column)]
        _draw_axonometric_scene(
            axis,
            _as_base_scene(scenes[panel.scene_key]),
            panel.method,
            image_cache[panel.scene_key],
        )
        axis.set_title(
            panel.title,
            fontsize=8.0,
            color="#4C78A8" if panel.method == "M1" else "#D66B4F",
            pad=5,
        )
    _add_titles_and_labels(fig, query_id)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=230, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def save_multicity_axonometric_assets(
    query_id: str,
    retrieval: pd.DataFrame,
    scenes: Mapping[SceneKey, BlockScene],
    output_dir: str | Path,
    top_k: int = 5,
) -> Path:
    """Export text-free axonometric panels and a two-row comparison sheet.

    The query is repeated at the start of both rows so the composite can be
    placed directly in a paper figure. Individual PNG files are also saved for
    layouts assembled manually in illustration software.
    """

    from visualization import _render_block_axonometric

    plan = build_multicity_comparison_plan(query_id, retrieval, top_k)
    _check_scene_coverage(plan, scenes)
    output_dir = Path(output_dir)
    panel_dir = output_dir / "panels"
    panel_dir.mkdir(parents=True, exist_ok=True)

    image_cache = {
        panel.scene_key: _render_block_axonometric(_as_base_scene(scenes[panel.scene_key]))
        for panel in plan
    }
    query = plan[0]
    plt.imsave(panel_dir / f"query_{query.city}_{query.block_id}.png", image_cache[query.scene_key])
    for panel in plan[1:]:
        plt.imsave(
            panel_dir
            / f"{panel.method.lower()}_{panel.rank:02d}_{panel.city}_{panel.block_id}.png",
            image_cache[panel.scene_key],
        )

    fig, axes = plt.subplots(
        2,
        top_k + 1,
        figsize=(15.2, 4.6),
        facecolor="white",
        gridspec_kw={"width_ratios": [1.15, *([1.0] * top_k)]},
    )
    for row in range(2):
        axes[row, 0].imshow(image_cache[query.scene_key])
        axes[row, 0].set_axis_off()
    for panel in plan[1:]:
        axis = axes[panel.row, panel.column]
        axis.imshow(image_cache[panel.scene_key])
        axis.set_axis_off()
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.025, hspace=0.08)
    composite_path = output_dir / f"{query_id}_axonometric_no_text.png"
    fig.savefig(
        composite_path,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0,
        facecolor="white",
    )
    plt.close(fig)
    return composite_path


def build_query_gain_cost(retrieval: pd.DataFrame, top_k: int = 5) -> pd.DataFrame:
    """Summarize roof gain, general-form cost and candidate replacement per query."""

    required = {
        "query_id",
        "method",
        "rank",
        "candidate_id",
        "candidate_city",
        "d_general",
        "d_roof",
    }
    _require_columns(retrieval, required, "retrieval")
    rows = []
    metadata = [
        column
        for column in (
            "query_role",
            "stratum",
            "query_category",
            "query_category_label",
            "query_selection_rank",
            "query_target_distance",
        )
        if column in retrieval
    ]
    for query_id, query_rows in retrieval.groupby("query_id", sort=False):
        methods = {}
        for method in ("M1", "M2"):
            frame = query_rows[query_rows["method"] == method].sort_values("rank")
            frame = frame[frame["rank"].astype(int).between(1, top_k)]
            if frame["rank"].astype(int).tolist() != list(range(1, top_k + 1)):
                raise ValueError(f"{query_id} {method} must contain ranks 1..{top_k}")
            methods[method] = frame
        m1_keys = set(zip(methods["M1"]["candidate_city"], methods["M1"]["candidate_id"]))
        m2_keys = set(zip(methods["M2"]["candidate_city"], methods["M2"]["candidate_id"]))
        overlap = len(m1_keys & m2_keys)
        roof_gain = float(
            methods["M1"]["d_roof"].mean() - methods["M2"]["d_roof"].mean()
        )
        general_cost = float(
            methods["M2"]["d_general"].mean()
            - methods["M1"]["d_general"].mean()
        )
        if np.isclose(roof_gain, 0.0, atol=1e-12):
            roof_gain = 0.0
        if np.isclose(general_cost, 0.0, atol=1e-12):
            general_cost = 0.0
        row = {
            "query_id": str(query_id),
            "roof_gain": roof_gain,
            "general_cost": general_cost,
            "top_k_overlap": overlap,
            "replacement_count": top_k - overlap,
            "m1_general_mean": float(methods["M1"]["d_general"].mean()),
            "m2_general_mean": float(methods["M2"]["d_general"].mean()),
            "m1_roof_mean": float(methods["M1"]["d_roof"].mean()),
            "m2_roof_mean": float(methods["M2"]["d_roof"].mean()),
        }
        for column in metadata:
            values = query_rows[column].dropna().astype(str).unique()
            row[column] = values[0] if len(values) else ""
        rows.append(row)
    return pd.DataFrame(rows).reset_index(drop=True)


def select_category_examples(
    diagnostics: pd.DataFrame, per_category: int = 1
) -> pd.DataFrame:
    """Choose queries nearest each category's median observed response."""

    if per_category <= 0:
        raise ValueError("per_category must be positive")
    required = {
        "query_id",
        "query_category",
        "roof_gain",
        "general_cost",
        "replacement_count",
    }
    _require_columns(diagnostics, required, "diagnostics")
    response_columns = ["roof_gain", "general_cost", "replacement_count"]
    selected = []
    for _, category in diagnostics.groupby("query_category", sort=False):
        category = category.copy()
        values = category[response_columns].astype(float)
        medians = values.median()
        scales = values.quantile(0.75) - values.quantile(0.25)
        scales = scales.mask((~np.isfinite(scales)) | (scales == 0), 1.0)
        category["example_distance"] = np.sqrt(
            np.mean(np.square((values - medians) / scales), axis=1)
        )
        category = category.sort_values(
            ["example_distance", "query_id"], kind="mergesort"
        ).head(per_category)
        category["example_role"] = "category_representative"
        selected.append(category)
    return pd.concat(selected, ignore_index=True)


def rank_query_effects(diagnostics: pd.DataFrame) -> pd.DataFrame:
    """Rank queries by equal-percentile roof gain and form preservation."""

    required = {"query_id", "roof_gain", "general_cost"}
    _require_columns(diagnostics, required, "diagnostics")
    if diagnostics.empty:
        raise ValueError("diagnostics cannot be empty")
    result = diagnostics.copy()
    count = len(result)

    def rank_zero_to_one(values: pd.Series, ascending: bool) -> pd.Series:
        if count == 1:
            return pd.Series(1.0, index=values.index)
        ranks = values.astype(float).rank(method="average", ascending=ascending)
        return (ranks - 1.0) / (count - 1.0)

    result["roof_benefit_percentile"] = rank_zero_to_one(
        result["roof_gain"], ascending=True
    )
    result["form_preservation_percentile"] = rank_zero_to_one(
        result["general_cost"], ascending=False
    )
    result["effect_balance_score"] = (
        result["roof_benefit_percentile"]
        + result["form_preservation_percentile"]
    ) / 2.0
    result = result.sort_values(
        ["effect_balance_score", "roof_gain", "general_cost", "query_id"],
        ascending=[False, False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    result.insert(0, "effect_rank", np.arange(1, len(result) + 1))
    return result


def make_query_gain_cost_figure(
    diagnostics: pd.DataFrame, output_path: str | Path | None = None
) -> plt.Figure:
    """Plot roof-distance improvement against loss of general-form similarity."""

    _require_columns(
        diagnostics,
        {"query_id", "roof_gain", "general_cost", "replacement_count"},
        "diagnostics",
    )
    fig, axis = plt.subplots(figsize=(10.8, 7.2), facecolor="white")
    sizes = 55 + diagnostics["replacement_count"].to_numpy(dtype=float) * 22
    has_categories = "query_category" in diagnostics.columns
    if has_categories:
        palette = {
            "near_flat": "#59A14F",
            "typical": "#4C78A8",
            "steep": "#E15759",
            "high_relief": "#B279A2",
            "heterogeneous": "#F28E2B",
        }
        fallback_colors = plt.get_cmap("tab10").colors
        for category_index, (category_name, group) in enumerate(
            diagnostics.groupby("query_category", sort=False)
        ):
            positions = diagnostics.index.get_indexer(group.index)
            label = (
                str(group["query_category_label"].iloc[0])
                if "query_category_label" in group
                else _display_city(category_name)
            )
            axis.scatter(
                group["general_cost"],
                group["roof_gain"],
                s=sizes[positions],
                color=palette.get(str(category_name), fallback_colors[category_index % 10]),
                label=label,
                alpha=0.86,
                edgecolor="white",
                linewidth=0.8,
                zorder=3,
            )
        axis.legend(frameon=False, loc="upper left", title="Roof category")
    else:
        roles = diagnostics.get(
            "query_role", pd.Series("regular", index=diagnostics.index)
        )
        boundary = roles.astype(str).eq("boundary").to_numpy()
        colors = np.where(boundary, "#D66B4F", "#4C78A8")
        axis.scatter(
            diagnostics["general_cost"],
            diagnostics["roof_gain"],
            s=sizes,
            c=colors,
            alpha=0.86,
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
    for row in diagnostics.itertuples(index=False):
        axis.annotate(
            str(row.query_id).replace("LU_LU_", ""),
            (row.general_cost, row.roof_gain),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
            color="#333333",
        )
    axis.axhline(0, color="#888888", linewidth=0.9)
    axis.axvline(0, color="#888888", linewidth=0.9)
    axis.fill_between(
        axis.get_xlim(), 0, axis.get_ylim()[1], color="#EAF3EF", alpha=0.34, zorder=0
    )
    axis.set_xlabel("General-form cost  (M2 mean distance − M1 mean distance)")
    axis.set_ylabel("Roof-similarity gain  (M1 mean distance − M2 mean distance)")
    axis.set_title(
        "What changes after adding roof information?",
        loc="left",
        fontsize=14,
        fontweight="bold",
    )
    axis.text(
        0.99,
        0.02,
        (
            "Larger circles = more top-5 candidates replaced"
            if has_categories
            else "Larger circles = more top-5 candidates replaced\nOrange = boundary query"
        ),
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color="#555555",
    )
    axis.grid(color="#ECECEC", linewidth=0.8)
    axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=240, bbox_inches="tight", facecolor="white")
    return fig


def make_candidate_city_shift_figure(
    retrieval: pd.DataFrame, output_path: str | Path | None = None
) -> plt.Figure:
    """Show which cities gain or lose top-k appearances from M1 to M2."""

    _require_columns(retrieval, {"method", "candidate_city"}, "retrieval")
    counts = (
        retrieval.groupby(["candidate_city", "method"]).size().unstack(fill_value=0)
    )
    for method in ("M1", "M2"):
        if method not in counts:
            counts[method] = 0
    counts["change"] = counts["M2"] - counts["M1"]
    counts = counts.sort_values(["change", "M2"], ascending=[True, True])
    y = np.arange(len(counts))
    fig_height = max(5.8, len(counts) * 0.40)
    fig, axis = plt.subplots(figsize=(10.5, fig_height), facecolor="white")
    for index, (_, row) in enumerate(counts.iterrows()):
        axis.plot([row["M1"], row["M2"]], [index, index], color="#D5D5D5", linewidth=2)
    axis.scatter(counts["M1"], y, color="#4C78A8", s=48, label="M1 baseline", zorder=3)
    axis.scatter(counts["M2"], y, color="#D66B4F", s=48, label="M2 roof-aware", zorder=3)
    axis.set_yticks(y, [_display_city(city) for city in counts.index])
    axis.set_xlabel("Appearances among all reported top-5 candidates")
    axis.set_title("Candidate-city composition changes", loc="left", fontsize=14, fontweight="bold")
    axis.grid(axis="x", color="#ECECEC", linewidth=0.8)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=240, bbox_inches="tight", facecolor="white")
    return fig


def make_city_roof_profile_figure(
    city_summary: pd.DataFrame,
    city_feature_medians: pd.DataFrame,
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Compare sample size and three interpretable city-level roof medians."""

    summary = city_summary.copy()
    medians = city_feature_medians.copy()
    if "city" not in medians.columns:
        medians = medians.reset_index().rename(columns={medians.index.name or "index": "city"})
    required_metrics = [
        "roof_mean_slope",
        "roof_relief_ratio_median",
        "building_roof_slope_sd",
    ]
    _require_columns(summary, {"city", "selected_blocks"}, "city_summary")
    _require_columns(medians, {"city", *required_metrics}, "city_feature_medians")
    data = summary[["city", "selected_blocks"]].merge(
        medians[["city", *required_metrics]], on="city", how="inner"
    )
    data = data.sort_values("roof_mean_slope", ascending=True).reset_index(drop=True)
    y = np.arange(len(data))
    fig, axes = plt.subplots(1, 4, figsize=(16, max(6.5, len(data) * 0.39)), sharey=True)
    specifications = [
        ("selected_blocks", "Selected blocks", "#79706E"),
        ("roof_mean_slope", "Median roof slope (°)", "#D66B4F"),
        ("roof_relief_ratio_median", "Median roof relief ratio", "#B279A2"),
        ("building_roof_slope_sd", "Median within-block slope SD (°)", "#59A14F"),
    ]
    for axis, (column, title, color) in zip(axes, specifications):
        axis.barh(y, data[column], color=color, alpha=0.82, height=0.64)
        axis.set_title(title, loc="left", fontsize=10.5, fontweight="bold")
        axis.grid(axis="x", color="#ECECEC", linewidth=0.8)
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.tick_params(axis="y", length=0)
    axes[0].set_yticks(y, [_display_city(city) for city in data["city"]])
    fig.suptitle(
        "Multicity sample and roof-profile medians",
        x=0.01,
        ha="left",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=240, bbox_inches="tight", facecolor="white")
    return fig
