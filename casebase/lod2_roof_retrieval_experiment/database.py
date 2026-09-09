"""Database boundary for the isolated Luxembourg roof experiment."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from shapely import wkb
from shapely.geometry import Polygon
from dotenv import load_dotenv


EXPERIMENT_SCHEMA = "exp_lod2_roof_lu"
BUILDING_TABLE = f"{EXPERIMENT_SCHEMA}.building_metrics"
BLOCK_TABLE = f"{EXPERIMENT_SCHEMA}.block_metrics"
RETRIEVAL_TABLE = f"{EXPERIMENT_SCHEMA}.retrieval_results"
FORMAL_QUERY_TABLE = f"{EXPERIMENT_SCHEMA}.formal_queries"
FORMAL_EVALUATION_TABLE = f"{EXPERIMENT_SCHEMA}.formal_evaluation_results"

SOURCE_BUILDING_TABLE = "lod2.luxembourg_buildings_lod2"
SOURCE_SURFACE_TABLE = "lod2.luxembourg_building_surfaces_lod2"
SOURCE_BLOCK_TABLE = "block.luxembourg_blocks"


@dataclass(frozen=True)
class BuildingGeometryRecord:
    building_id: str
    block_id: str
    source_id: str | None
    source_height: float | None
    footprint: Polygon
    roof_polygons: tuple[Polygon, ...]
    ground_polygons: tuple[Polygon, ...]


def assert_experiment_relation(relation: str) -> str:
    """Reject every write target outside the dedicated experiment schema."""

    allowed = {
        BUILDING_TABLE,
        BLOCK_TABLE,
        RETRIEVAL_TABLE,
        FORMAL_QUERY_TABLE,
        FORMAL_EVALUATION_TABLE,
    }
    if relation not in allowed:
        raise ValueError(f"Database writes are restricted to {sorted(allowed)}")
    return relation


def _find_env_path(start: str | Path | None = None) -> Path:
    current = Path(start or Path.cwd()).resolve()
    if current.is_file():
        return current
    for directory in (current, *current.parents):
        candidate = directory / ".env"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("Could not find a project .env file; pass env_path explicitly")


def connect_from_env(env_path: str | Path | None = None):
    """Open a psycopg2 connection using the project-style DB_* variables."""

    load_dotenv(_find_env_path(env_path), override=False)
    return psycopg2.connect(
        host=os.getenv("DB_HOST"),
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        port=int(os.getenv("DB_PORT", "5432")),
    )


def fetch_lod2_metric_srid(connection) -> int:
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT lod2_metric_srid
            FROM summary.city_name_map
            WHERE city_query = %s
            """,
            ("luxembourg",),
        )
        row = cursor.fetchone()
    if not row or not row[0]:
        raise RuntimeError("Luxembourg lod2_metric_srid is not configured")
    return int(row[0])


def _decode_polygon(value) -> Polygon:
    polygon = wkb.loads(bytes(value))
    if not isinstance(polygon, Polygon):
        raise ValueError(f"Expected Polygon geometry, got {polygon.geom_type}")
    return polygon


def iter_luxembourg_buildings(
    connection, metric_srid: int, block_ids: Sequence[str] | None = None
) -> Iterator[BuildingGeometryRecord]:
    """Stream matched buildings grouped with all roof and ground surfaces."""

    selected_blocks = sorted({str(value) for value in block_ids}) if block_ids else None
    block_filter = "AND b.block_id = ANY(%s)" if selected_blocks else ""
    params = (metric_srid, metric_srid, selected_blocks) if selected_blocks else (
        metric_srid,
        metric_srid,
    )
    cursor = connection.cursor(name="lod2_roof_lu_buildings")
    cursor.itersize = 2000
    cursor.execute(
        f"""
        SELECT b.building_id, b.block_id, b.citygml_id, b.height,
               ST_AsEWKB(ST_Transform(b.geom_2d, %s)) AS footprint_wkb,
               s.surface_type,
               ST_AsEWKB(ST_Transform(s.geom_3d, %s)) AS surface_wkb
        FROM {SOURCE_BUILDING_TABLE} AS b
        JOIN {SOURCE_SURFACE_TABLE} AS s ON s.building_id = b.building_id
        WHERE b.block_id IS NOT NULL
          AND s.surface_type IN ('RoofSurface', 'GroundSurface')
          {block_filter}
        ORDER BY b.building_id, s.surface_type, s.surface_id
        """,
        params,
    )
    current_id = None
    current = None
    roofs: list[Polygon] = []
    grounds: list[Polygon] = []
    try:
        for row in cursor:
            building_id, block_id, source_id, source_height, footprint_wkb, surface_type, surface_wkb = row
            if current_id is not None and building_id != current_id:
                yield BuildingGeometryRecord(
                    building_id=current[0],
                    block_id=current[1],
                    source_id=current[2],
                    source_height=current[3],
                    footprint=current[4],
                    roof_polygons=tuple(roofs),
                    ground_polygons=tuple(grounds),
                )
                roofs, grounds = [], []
            if building_id != current_id:
                current_id = building_id
                current = (
                    str(building_id),
                    str(block_id),
                    source_id,
                    float(source_height) if source_height is not None else None,
                    _decode_polygon(footprint_wkb),
                )
            polygon = _decode_polygon(surface_wkb)
            if surface_type == "RoofSurface":
                roofs.append(polygon)
            else:
                grounds.append(polygon)
        if current_id is not None and current is not None:
            yield BuildingGeometryRecord(
                building_id=current[0],
                block_id=current[1],
                source_id=current[2],
                source_height=current[3],
                footprint=current[4],
                roof_polygons=tuple(roofs),
                ground_polygons=tuple(grounds),
            )
    finally:
        cursor.close()


def fetch_luxembourg_blocks(connection, metric_srid: int) -> pd.DataFrame:
    """Read blocks containing matched Luxembourg LOD2 buildings."""

    with connection.cursor() as cursor:
        cursor.execute(
            f"""
            SELECT bl.block_id, bl.area_m2,
                   ST_Perimeter(ST_Transform(bl.geom, %s)) AS perimeter_m,
                   ST_AsEWKB(ST_Transform(bl.geom, %s)) AS geom_wkb
            FROM {SOURCE_BLOCK_TABLE} AS bl
            WHERE EXISTS (
                SELECT 1 FROM {SOURCE_BUILDING_TABLE} AS b
                WHERE b.block_id = bl.block_id
            )
            ORDER BY bl.block_id
            """,
            (metric_srid, metric_srid),
        )
        rows = cursor.fetchall()
    return pd.DataFrame(
        [
            {
                "block_id": str(block_id),
                "area_m2": float(area_m2),
                "perimeter_m": float(perimeter_m),
                "geometry": _decode_polygon(geom_wkb),
            }
            for block_id, area_m2, perimeter_m, geom_wkb in rows
        ]
    )


def ensure_experiment_schema(connection) -> None:
    """Create only the isolated schema and its five result tables."""

    statements = [
        f"CREATE SCHEMA IF NOT EXISTS {EXPERIMENT_SCHEMA}",
        f"""
        CREATE TABLE IF NOT EXISTS {BUILDING_TABLE} (
            building_id TEXT PRIMARY KEY, block_id TEXT NOT NULL,
            footprint_area DOUBLE PRECISION, centroid_x DOUBLE PRECISION,
            centroid_y DOUBLE PRECISION, orientation_deg DOUBLE PRECISION,
            h_g1 DOUBLE PRECISION, roof_area_3d DOUBLE PRECISION,
            roof_projected_area DOUBLE PRECISION,
            roof_mean_slope DOUBLE PRECISION, roof_relief_ratio DOUBLE PRECISION,
            roof_surface_count INTEGER, ground_surface_count INTEGER,
            valid_triangle_count INTEGER, is_usable BOOLEAN NOT NULL,
            quality_codes TEXT NOT NULL
        )
        """,
        f"""
        CREATE TABLE IF NOT EXISTS {BLOCK_TABLE} (
            block_id TEXT PRIMARY KEY, area_m2 DOUBLE PRECISION,
            compactness DOUBLE PRECISION, elongation DOUBLE PRECISION,
            bcr DOUBLE PRECISION, volume_density DOUBLE PRECISION,
            mean_h_g1 DOUBLE PRECISION, h_g1_cv DOUBLE PRECISION,
            building_count INTEGER, log_block_area DOUBLE PRECISION,
            log_building_count DOUBLE PRECISION,
            mean_footprint_area DOUBLE PRECISION,
            footprint_area_cv DOUBLE PRECISION,
            orientation_entropy DOUBLE PRECISION,
            normalized_nn_distance DOUBLE PRECISION,
            roof_mean_slope DOUBLE PRECISION,
            roof_relief_ratio_median DOUBLE PRECISION,
            building_roof_slope_sd DOUBLE PRECISION,
            usable_footprint_ratio DOUBLE PRECISION,
            is_usable BOOLEAN NOT NULL, quality_codes TEXT NOT NULL
        )
        """,
        f"""
        CREATE TABLE IF NOT EXISTS {RETRIEVAL_TABLE} (
            query_id TEXT NOT NULL, method TEXT NOT NULL,
            lambda_value DOUBLE PRECISION NOT NULL, rank INTEGER NOT NULL,
            candidate_id TEXT NOT NULL, d_general DOUBLE PRECISION NOT NULL,
            d_roof DOUBLE PRECISION NOT NULL, d_combined DOUBLE PRECISION NOT NULL,
            pool_size INTEGER NOT NULL,
            PRIMARY KEY (query_id, method, lambda_value, rank)
        )
        """,
        f"""
        CREATE TABLE IF NOT EXISTS {FORMAL_QUERY_TABLE} (
            block_id TEXT PRIMARY KEY, query_role TEXT NOT NULL,
            stratum TEXT NOT NULL, roof_mean_slope DOUBLE PRECISION NOT NULL,
            roof_relief_ratio_median DOUBLE PRECISION NOT NULL,
            building_roof_slope_sd DOUBLE PRECISION NOT NULL,
            selection_seed INTEGER NOT NULL
        )
        """,
        f"""
        CREATE TABLE IF NOT EXISTS {FORMAL_EVALUATION_TABLE} (
            query_id TEXT NOT NULL, query_role TEXT NOT NULL,
            stratum TEXT NOT NULL, method TEXT NOT NULL,
            lambda_value DOUBLE PRECISION NOT NULL, rank INTEGER NOT NULL,
            candidate_id TEXT NOT NULL, d_general DOUBLE PRECISION NOT NULL,
            d_roof DOUBLE PRECISION NOT NULL, d_combined DOUBLE PRECISION NOT NULL,
            pool_size INTEGER NOT NULL, general_pool_rank INTEGER NOT NULL,
            slope_wasserstein_deg DOUBLE PRECISION NOT NULL,
            relief_wasserstein DOUBLE PRECISION NOT NULL,
            flat_roof_share_abs_diff DOUBLE PRECISION NOT NULL,
            PRIMARY KEY (query_id, method, lambda_value, rank)
        )
        """,
    ]
    with connection.cursor() as cursor:
        for statement in statements:
            cursor.execute(statement)
    connection.commit()


_BUILDING_COLUMNS = [
    "building_id", "block_id", "footprint_area", "centroid_x", "centroid_y",
    "orientation_deg", "h_g1", "roof_area_3d", "roof_projected_area",
    "roof_mean_slope", "roof_relief_ratio", "roof_surface_count",
    "ground_surface_count", "valid_triangle_count", "is_usable", "quality_codes",
]
_BLOCK_COLUMNS = [
    "block_id", "area_m2", "compactness", "elongation", "bcr",
    "volume_density", "mean_h_g1", "h_g1_cv", "building_count",
    "log_block_area", "log_building_count", "mean_footprint_area",
    "footprint_area_cv", "orientation_entropy", "normalized_nn_distance",
    "roof_mean_slope", "roof_relief_ratio_median", "building_roof_slope_sd",
    "usable_footprint_ratio", "is_usable", "quality_codes",
]
_RETRIEVAL_COLUMNS = [
    "query_id", "method", "lambda_value", "rank", "candidate_id",
    "d_general", "d_roof", "d_combined", "pool_size",
]
_FORMAL_QUERY_COLUMNS = [
    "block_id", "query_role", "stratum", "roof_mean_slope",
    "roof_relief_ratio_median", "building_roof_slope_sd", "selection_seed",
]
_FORMAL_EVALUATION_COLUMNS = [
    "query_id", "query_role", "stratum", "method", "lambda_value", "rank",
    "candidate_id", "d_general", "d_roof", "d_combined", "pool_size",
    "general_pool_rank", "slope_wasserstein_deg", "relief_wasserstein",
    "flat_roof_share_abs_diff",
]


def _clean_value(value):
    if value is None or (isinstance(value, (float, np.floating)) and not np.isfinite(value)):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def _replace_frame(connection, relation: str, frame: pd.DataFrame, columns: Sequence[str]) -> None:
    assert_experiment_relation(relation)
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing columns for {relation}: {missing}")
    rows = [tuple(_clean_value(value) for value in row) for row in frame[list(columns)].itertuples(index=False, name=None)]
    with connection.cursor() as cursor:
        cursor.execute(f"TRUNCATE TABLE {relation}")
        if rows:
            column_sql = ", ".join(columns)
            execute_values(
                cursor,
                f"INSERT INTO {relation} ({column_sql}) VALUES %s",
                rows,
                page_size=1000,
            )
    connection.commit()


def replace_building_metrics(connection, frame: pd.DataFrame) -> None:
    _replace_frame(connection, BUILDING_TABLE, frame, _BUILDING_COLUMNS)


def replace_block_metrics(connection, frame: pd.DataFrame) -> None:
    _replace_frame(connection, BLOCK_TABLE, frame, _BLOCK_COLUMNS)


def replace_retrieval_results(connection, frame: pd.DataFrame) -> None:
    _replace_frame(connection, RETRIEVAL_TABLE, frame, _RETRIEVAL_COLUMNS)


def replace_formal_queries(connection, frame: pd.DataFrame) -> None:
    _replace_frame(connection, FORMAL_QUERY_TABLE, frame, _FORMAL_QUERY_COLUMNS)


def replace_formal_evaluation_results(connection, frame: pd.DataFrame) -> None:
    _replace_frame(
        connection,
        FORMAL_EVALUATION_TABLE,
        frame,
        _FORMAL_EVALUATION_COLUMNS,
    )
