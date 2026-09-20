"""Limited phase 3 contact sheets and audit distributions."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from shapely import wkt
from shapely.geometry import MultiPolygon, Polygon

from audit_phase3 import _write_csv
from export_phase3 import read_graph


PHASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PHASE_DIR / "results"
VISUAL_DIR = RESULTS_DIR / "visual_checks"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def select_city_anchors(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    selected = []
    for city in sorted({row["city"] for row in rows}):
        eligible = [row for row in rows if row["city"] == city and "core_audit" in row["sample_groups"]]
        counts = np.asarray([float(row["node_count"]) for row in eligible])
        median = float(np.median(counts))
        choice = min(eligible, key=lambda row: (abs(float(row["node_count"]) - median), row["sample_id"]))
        selected.append({**choice, "selection_reason": "closest core sample to city median node count"})
    return selected


def select_flagged_cases(rows: list[dict[str, str]], status: list[dict[str, str]]) -> list[dict[str, str]]:
    status_by_id = {row["sample_id"]: row for row in status}
    selected: dict[str, dict[str, str]] = {}

    def add(candidates: list[dict[str, str]], reason: str, limit: int) -> None:
        for row in candidates[:limit]:
            if row["sample_id"] not in selected:
                selected[row["sample_id"]] = {**row, "selection_reason": reason}

    add(sorted((row for row in rows if float(row["cell_failure_count"]) > 0), key=lambda row: (-float(row["cell_failure_count"]), row["sample_id"])), "cell failures", 3)
    add(sorted(rows, key=lambda row: (-float(row["combined_weak_candidate_ratio"]), row["sample_id"])), "high combined weak-edge ratio", 3)
    add(sorted((row for row in rows if row.get("bcr") and float(row["bcr"]) > 1), key=lambda row: (-float(row["bcr"]), row["sample_id"])), "BCR greater than 1", 2)
    boundary = [row for row in rows if "boundary" in status_by_id[row["sample_id"]].get("quality_flags", "")]
    add(sorted(boundary, key=lambda row: row["sample_id"]), "building footprint boundary quality flag", 2)
    runtime = sorted(rows, key=lambda row: (-float(status_by_id[row["sample_id"]]["runtime_seconds"]), row["sample_id"]))
    add(runtime, "long runtime", 2)
    return list(selected.values())[:12]


def _parts(geometry):
    if isinstance(geometry, Polygon):
        return [geometry]
    if isinstance(geometry, MultiPolygon):
        return list(geometry.geoms)
    return [part for part in getattr(geometry, "geoms", []) if isinstance(part, Polygon)]


def _outline(axis, geometry, **kwargs):
    for polygon in _parts(geometry):
        x, y = polygon.exterior.xy
        axis.plot(x, y, **kwargs)


def _draw_graph(axis, record: dict[str, Any], title: str) -> None:
    block = wkt.loads(record["graph"]["block_geometry_wkt_metric"])
    _outline(axis, block, color="#111827", linewidth=1.1)
    nodes = {str(node["building_id"]): node for node in record["nodes"]}
    for cell in record["cells"]:
        _outline(axis, wkt.loads(cell["geometry_wkt_metric"]), color="#93c5fd", linewidth=0.25, alpha=0.65)
    for edge in record["edges"]:
        first, second = nodes[str(edge["source"])], nodes[str(edge["target"])]
        color = "#dc2626" if edge["combined_weak_candidate"] else "#f59e0b" if edge["long_distance_candidate"] else "#6b7280"
        axis.plot(
            [first["representative_point_x_m"], second["representative_point_x_m"]],
            [first["representative_point_y_m"], second["representative_point_y_m"]],
            color=color,
            linewidth=0.35,
            alpha=0.7,
        )
    for node in record["nodes"]:
        for polygon in _parts(wkt.loads(node["geometry_wkt_metric"])):
            x, y = polygon.exterior.xy
            axis.fill(x, y, color="#374151", alpha=0.9, linewidth=0)
    axis.set_aspect("equal", adjustable="box")
    axis.axis("off")
    axis.set_title(title, fontsize=8)


def _contact_sheet(rows: list[dict[str, str]], path: Path, columns: int, title: str) -> None:
    rows_count = int(np.ceil(len(rows) / columns))
    figure, axes = plt.subplots(rows_count, columns, figsize=(columns * 3.2, rows_count * 3.0))
    axes = np.atleast_1d(axes).ravel()
    for axis, row in zip(axes, rows):
        record = read_graph(RESULTS_DIR / "dataset" / "graphs" / f"{row['sample_id']}.graph.json.gz")
        _draw_graph(axis, record, f"{row['city']}\n{row['block_id']}  n={row['node_count']} e={row['edge_count']}")
    for axis in axes[len(rows):]:
        axis.axis("off")
    figure.suptitle(title, fontsize=15)
    figure.tight_layout(rect=(0, 0, 1, 0.98))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _distribution_plots(rows: list[dict[str, str]], status: list[dict[str, str]]) -> None:
    dist_dir = VISUAL_DIR / "distributions"
    dist_dir.mkdir(parents=True, exist_ok=True)
    status_by_id = {row["sample_id"]: row for row in status}

    figure, axes = plt.subplots(2, 2, figsize=(10, 8))
    for axis, field, label in zip(
        axes.ravel(),
        ("node_count", "edge_count", "average_degree", "connected_components"),
        ("Nodes", "Edges", "Average degree", "Connected components"),
    ):
        axis.hist([float(row[field]) for row in rows], bins=35, color="#2563eb", alpha=0.8)
        axis.set_title(label)
    figure.tight_layout()
    figure.savefig(dist_dir / "graph_structure_distributions.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for axis, field, label in zip(
        axes,
        ("cell_failure_count", "long_distance_candidate_ratio", "combined_weak_candidate_ratio"),
        ("Cell failures", "Long-edge ratio", "Combined weak ratio"),
    ):
        axis.hist([float(row[field]) for row in rows], bins=30, color="#d97706", alpha=0.8)
        axis.set_title(label)
    figure.tight_layout()
    figure.savefig(dist_dir / "quality_distributions.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    x = np.asarray([float(row["node_count"]) for row in rows])
    runtime = np.asarray([float(status_by_id[row["sample_id"]]["runtime_seconds"]) for row in rows])
    size_mb = np.asarray([float(status_by_id[row["sample_id"]]["output_bytes"]) / 1024**2 for row in rows])
    axes[0].scatter(x, runtime, s=9, alpha=0.5)
    axes[0].set(xlabel="Nodes", ylabel="Runtime (s)", title="Runtime vs nodes")
    axes[1].scatter(x, size_mb, s=9, alpha=0.5, color="#059669")
    axes[1].set(xlabel="Nodes", ylabel="Compressed graph (MiB)", title="Storage vs nodes")
    figure.tight_layout()
    figure.savefig(dist_dir / "runtime_storage.png", dpi=180)
    plt.close(figure)

    profiles = _read_csv(RESULTS_DIR / "audit" / "feature_profile_global.csv")
    registry = _read_csv(RESULTS_DIR / "feature_role_registry_v0.3.csv")
    core = {row["field"] for row in registry if row["role"] == "core_model"}
    missing = sorted(
        ((row["field"], float(row["missing_ratio"])) for row in profiles if row["field"] in core),
        key=lambda item: item[1],
        reverse=True,
    )
    figure, axis = plt.subplots(figsize=(10, max(4, len(missing) * 0.22)))
    axis.barh([item[0] for item in missing][::-1], [item[1] for item in missing][::-1], color="#7c3aed")
    axis.set(xlabel="Missing ratio", title="Core candidate feature missingness")
    figure.tight_layout()
    figure.savefig(dist_dir / "core_feature_missingness.png", dpi=180)
    plt.close(figure)


def main() -> None:
    rows = _read_csv(RESULTS_DIR / "audit" / "graph_structure_summary.csv")
    status = _read_csv(RESULTS_DIR / "audit" / "encoding_status.csv")
    city_anchors = select_city_anchors(rows)
    flagged = select_flagged_cases(rows, status)
    _write_csv(VISUAL_DIR / "city_anchor_selection.csv", city_anchors)
    _write_csv(VISUAL_DIR / "flagged_case_selection.csv", flagged)
    _contact_sheet(city_anchors, VISUAL_DIR / "city_anchors_contact_sheet.png", 4, "Phase 3 city anchors")
    _contact_sheet(flagged, VISUAL_DIR / "flagged_cases_contact_sheet.png", 4, "Phase 3 flagged cases")
    _distribution_plots(rows, status)
    print({"city_anchors": len(city_anchors), "flagged_cases": len(flagged)})


if __name__ == "__main__":
    main()
