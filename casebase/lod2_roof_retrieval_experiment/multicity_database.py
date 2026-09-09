"""Database boundary for the isolated multicity LOD2 roof experiment."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterator, Sequence

import numpy as np
import pandas as pd
from psycopg2.extras import execute_values
from shapely import wkb
from shapely.errors import GEOSException
from shapely.geometry import MultiPolygon, Polygon

from database import BuildingGeometryRecord, connect_from_env


EXPERIMENT_SCHEMA = "exp_lod2_roof_multi"
SAMPLE_TABLE = f"{EXPERIMENT_SCHEMA}.sample_manifest"
BUILDING_TABLE = f"{EXPERIMENT_SCHEMA}.building_metrics"
BLOCK_TABLE = f"{EXPERIMENT_SCHEMA}.block_metrics"
RETRIEVAL_TABLE = f"{EXPERIMENT_SCHEMA}.retrieval_results"

_SAFE_CITY = re.compile(r"^[a-z][a-z0-9_]*$")


@dataclass(frozen=True)
class CitySource:
    city: str
    display_name: str
    metric_srid: int

    def __post_init__(self) -> None:
        normalized = str(self.city).strip().lower()
        if not _SAFE_CITY.fullmatch(normalized):
            raise ValueError(f"Unsafe city catalog identifier: {self.city!r}")
        if not str(self.display_name).strip():
            raise ValueError("display_name cannot be empty")
        if int(self.metric_srid) <= 0:
            raise ValueError("metric_srid must be positive")
        object.__setattr__(self, "city", normalized)
        object.__setattr__(self, "display_name", str(self.display_name).strip())
        object.__setattr__(self, "metric_srid", int(self.metric_srid))

    @property
    def block_table(self) -> str:
        return f"block.{self.city}_blocks"

    @property
    def building_table(self) -> str:
        return f"lod2.{self.city}_buildings_lod2"

    @property
    def surface_table(self) -> str:
        return f"lod2.{self.city}_building_surfaces_lod2"


def discover_lod2_cities(connection) -> list[CitySource]:
    """Read every city with an explicit LOD2 metric CRS from the catalog."""

    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT city_query, city, lod2_metric_srid
            FROM summary.city_name_map
            WHERE lod2_metric_srid IS NOT NULL
            ORDER BY city_query
            """
        )
        rows = cursor.fetchall()
    return [CitySource(city, display_name, metric_srid) for city, display_name, metric_srid in rows]


def validate_source_columns(connection, source: CitySource) -> None:
    """Fail before processing when a standardized source relation is incomplete."""

    requirements = {
        ("block", f"{source.city}_blocks"): {"block_id", "geom"},
        ("lod2", f"{source.city}_buildings_lod2"): {
            "building_id",
            "block_id",
            "citygml_id",
            "height",
            "geom_2d",
        },
        ("lod2", f"{source.city}_building_surfaces_lod2"): {
            "surface_id",
            "building_id",
            "surface_type",
            "geom_3d",
        },
    }
    with connection.cursor() as cursor:
        for (schema, table), required in requirements.items():
            cursor.execute(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = %s AND table_name = %s
                """,
                (schema, table),
            )
            available = {str(row[0]) for row in cursor.fetchall()}
            missing = sorted(required - available)
            if missing:
                raise RuntimeError(f"{schema}.{table} missing columns: {missing}")


def fetch_candidate_block_ids(connection, source: CitySource) -> list[str]:
    """Return all block IDs that have at least one matched LOD2 building."""

    with connection.cursor() as cursor:
        cursor.execute(
            f"""
            SELECT DISTINCT b.block_id
            FROM {source.building_table} AS b
            WHERE b.block_id IS NOT NULL
            ORDER BY b.block_id
            """
        )
        rows = cursor.fetchall()
    return [str(row[0]) for row in rows]


def fetch_luxembourg_formal_queries(connection) -> pd.DataFrame:
    """Read the frozen Luxembourg query IDs for cross-city retrieval."""

    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT block_id, query_role, stratum
            FROM exp_lod2_roof_lu.formal_queries
            ORDER BY query_role, stratum, block_id
            """
        )
        rows = cursor.fetchall()
    return pd.DataFrame(rows, columns=["block_id", "query_role", "stratum"])


def _decode_area_geometry(value):
    geometry = wkb.loads(bytes(value))
    if not isinstance(geometry, (Polygon, MultiPolygon)):
        raise ValueError(f"Expected Polygon or MultiPolygon, got {geometry.geom_type}")
    return geometry


def _decode_polygon_parts(value) -> tuple[Polygon, ...]:
    geometry = wkb.loads(bytes(value))
    if isinstance(geometry, Polygon):
        return (geometry,)
    if isinstance(geometry, MultiPolygon):
        return tuple(geometry.geoms)
    raise ValueError(f"Expected Polygon or MultiPolygon, got {geometry.geom_type}")


def fetch_city_blocks(
    connection, source: CitySource, block_ids: Sequence[str]
) -> pd.DataFrame:
    """Read a selected block batch in the city's metric CRS."""

    selected = sorted({str(value) for value in block_ids})
    columns = ["block_id", "area_m2", "perimeter_m", "geometry"]
    if not selected:
        return pd.DataFrame(columns=columns)
    with connection.cursor() as cursor:
        cursor.execute(
            f"""
            SELECT bl.block_id,
                   ST_Area(ST_Transform(bl.geom, %s)) AS area_m2,
                   ST_Perimeter(ST_Transform(bl.geom, %s)) AS perimeter_m,
                   ST_AsEWKB(ST_Transform(bl.geom, %s)) AS geom_wkb
            FROM {source.block_table} AS bl
            WHERE bl.block_id = ANY(%s)
            ORDER BY bl.block_id
            """,
            (source.metric_srid, source.metric_srid, source.metric_srid, selected),
        )
        rows = cursor.fetchall()
    return pd.DataFrame(
        [
            {
                "block_id": str(block_id),
                "area_m2": float(area_m2),
                "perimeter_m": float(perimeter_m),
                "geometry": _decode_area_geometry(geom_wkb),
            }
            for block_id, area_m2, perimeter_m, geom_wkb in rows
        ],
        columns=columns,
    )


def iter_city_buildings(
    connection, source: CitySource, block_ids: Sequence[str]
) -> Iterator[BuildingGeometryRecord]:
    """Stream all selected buildings, including those with no usable LOD2 surfaces."""

    selected = sorted({str(value) for value in block_ids})
    if not selected:
        return
    cursor = connection.cursor(name=f"lod2_roof_multi_{source.city}_buildings")
    cursor.itersize = 2000
    cursor.execute(
        f"""
        SELECT b.building_id, b.block_id, b.citygml_id, b.height,
               ST_AsEWKB(ST_Transform(b.geom_2d, %s)) AS footprint_wkb,
               s.surface_type,
               ST_AsEWKB(ST_Transform(s.geom_3d, %s)) AS surface_wkb
        FROM {source.building_table} AS b
        LEFT JOIN {source.surface_table} AS s
          ON s.building_id = b.building_id
         AND s.surface_type IN ('RoofSurface', 'GroundSurface')
        WHERE b.block_id = ANY(%s)
          AND b.geom_2d IS NOT NULL
        ORDER BY b.building_id, s.surface_type, s.surface_id
        """,
        (source.metric_srid, source.metric_srid, selected),
    )
    current_id = None
    current = None
    roofs: list[Polygon] = []
    grounds: list[Polygon] = []
    source_quality_codes: list[str] = []

    def make_record():
        return BuildingGeometryRecord(
            building_id=current[0],
            block_id=current[1],
            source_id=current[2],
            source_height=current[3],
            footprint=current[4],
            roof_polygons=tuple(roofs),
            ground_polygons=tuple(grounds),
            source_quality_codes=tuple(source_quality_codes),
        )

    try:
        for row in cursor:
            (
                building_id,
                block_id,
                source_id,
                source_height,
                footprint_wkb,
                surface_type,
                surface_wkb,
            ) = row
            if current_id is not None and building_id != current_id:
                yield make_record()
                roofs, grounds, source_quality_codes = [], [], []
            if building_id != current_id:
                current_id = building_id
                current = (
                    str(building_id),
                    str(block_id),
                    source_id,
                    float(source_height) if source_height is not None else None,
                    _decode_area_geometry(footprint_wkb),
                )
            if surface_type is None or surface_wkb is None:
                continue
            try:
                parts = _decode_polygon_parts(surface_wkb)
            except (GEOSException, TypeError, ValueError):
                if "invalid_surface_geometry" not in source_quality_codes:
                    source_quality_codes.append("invalid_surface_geometry")
                continue
            if surface_type == "RoofSurface":
                roofs.extend(parts)
            elif surface_type == "GroundSurface":
                grounds.extend(parts)
        if current_id is not None:
            yield make_record()
    finally:
        cursor.close()


def assert_multicity_relation(relation: str) -> str:
    allowed = {SAMPLE_TABLE, BUILDING_TABLE, BLOCK_TABLE, RETRIEVAL_TABLE}
    if relation not in allowed:
        raise ValueError(f"Database writes are restricted to {sorted(allowed)}")
    return relation


def ensure_multicity_schema(connection) -> None:
    """Create only the isolated multicity experiment schema and tables."""

    statements = [
        f"CREATE SCHEMA IF NOT EXISTS {EXPERIMENT_SCHEMA}",
        f"""
        CREATE TABLE IF NOT EXISTS {SAMPLE_TABLE} (
            city TEXT NOT NULL, block_id TEXT NOT NULL, random_order INTEGER NOT NULL,
            processed BOOLEAN NOT NULL, data_valid BOOLEAN,
            in_reference_scope BOOLEAN, is_eligible BOOLEAN,
            selected BOOLEAN NOT NULL, exclusion_codes TEXT NOT NULL,
            PRIMARY KEY (city, block_id)
        )
        """,
        f"""
        CREATE TABLE IF NOT EXISTS {BUILDING_TABLE} (
            city TEXT NOT NULL, building_id TEXT NOT NULL, block_id TEXT NOT NULL,
            footprint_area DOUBLE PRECISION, centroid_x DOUBLE PRECISION,
            centroid_y DOUBLE PRECISION, orientation_deg DOUBLE PRECISION,
            h_g1 DOUBLE PRECISION, roof_area_3d DOUBLE PRECISION,
            roof_projected_area DOUBLE PRECISION,
            roof_mean_slope DOUBLE PRECISION, roof_relief_ratio DOUBLE PRECISION,
            roof_surface_count INTEGER, ground_surface_count INTEGER,
            valid_triangle_count INTEGER, is_usable BOOLEAN NOT NULL,
            quality_codes TEXT NOT NULL,
            PRIMARY KEY (city, building_id)
        )
        """,
        f"""
        CREATE TABLE IF NOT EXISTS {BLOCK_TABLE} (
            city TEXT NOT NULL, block_id TEXT NOT NULL, area_m2 DOUBLE PRECISION,
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
            is_usable BOOLEAN NOT NULL, quality_codes TEXT NOT NULL,
            data_valid BOOLEAN NOT NULL, in_reference_scope BOOLEAN NOT NULL,
            is_eligible BOOLEAN NOT NULL, selected BOOLEAN NOT NULL,
            exclusion_codes TEXT NOT NULL,
            PRIMARY KEY (city, block_id)
        )
        """,
        f"""
        CREATE TABLE IF NOT EXISTS {RETRIEVAL_TABLE} (
            query_id TEXT NOT NULL, query_city TEXT NOT NULL,
            method TEXT NOT NULL, lambda_value DOUBLE PRECISION NOT NULL,
            rank INTEGER NOT NULL, candidate_id TEXT NOT NULL,
            candidate_city TEXT NOT NULL, same_city BOOLEAN NOT NULL,
            d_general DOUBLE PRECISION NOT NULL, d_roof DOUBLE PRECISION NOT NULL,
            d_combined DOUBLE PRECISION NOT NULL, pool_size INTEGER NOT NULL,
            general_pool_rank INTEGER NOT NULL,
            PRIMARY KEY (query_city, query_id, method, lambda_value, rank)
        )
        """,
    ]
    with connection.cursor() as cursor:
        for statement in statements:
            cursor.execute(statement)
    connection.commit()


_SAMPLE_COLUMNS = [
    "city", "block_id", "random_order", "processed", "data_valid",
    "in_reference_scope", "is_eligible", "selected", "exclusion_codes",
]
_BUILDING_COLUMNS = [
    "city", "building_id", "block_id", "footprint_area", "centroid_x",
    "centroid_y", "orientation_deg", "h_g1", "roof_area_3d",
    "roof_projected_area", "roof_mean_slope", "roof_relief_ratio",
    "roof_surface_count", "ground_surface_count", "valid_triangle_count",
    "is_usable", "quality_codes",
]
_BLOCK_COLUMNS = [
    "city", "block_id", "area_m2", "compactness", "elongation", "bcr",
    "volume_density", "mean_h_g1", "h_g1_cv", "building_count",
    "log_block_area", "log_building_count", "mean_footprint_area",
    "footprint_area_cv", "orientation_entropy", "normalized_nn_distance",
    "roof_mean_slope", "roof_relief_ratio_median", "building_roof_slope_sd",
    "usable_footprint_ratio", "is_usable", "quality_codes", "data_valid",
    "in_reference_scope", "is_eligible", "selected", "exclusion_codes",
]
_RETRIEVAL_COLUMNS = [
    "query_id", "query_city", "method", "lambda_value", "rank",
    "candidate_id", "candidate_city", "same_city", "d_general", "d_roof",
    "d_combined", "pool_size", "general_pool_rank",
]


def _clean_value(value):
    if value is pd.NA:
        return None
    if value is None or (
        isinstance(value, (float, np.floating)) and not np.isfinite(value)
    ):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def frame_value_batches(
    frame: pd.DataFrame,
    columns: Sequence[str],
    batch_size: int = 5000,
):
    """Yield database-ready rows without materialising the full frame as tuples."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    values = frame[list(columns)]
    for start in range(0, len(values), batch_size):
        batch = values.iloc[start : start + batch_size]
        yield [
            tuple(_clean_value(value) for value in row)
            for row in batch.itertuples(index=False, name=None)
        ]


def _replace_partition(
    connection,
    relation: str,
    frame: pd.DataFrame,
    columns: Sequence[str],
    partition_column: str,
    partition_value: str,
) -> None:
    try:
        with connection.cursor() as cursor:
            _replace_partition_in_cursor(
                cursor,
                relation,
                frame,
                columns,
                partition_column,
                partition_value,
            )
        connection.commit()
    except Exception:
        connection.rollback()
        raise


def _replace_partition_in_cursor(
    cursor,
    relation: str,
    frame: pd.DataFrame,
    columns: Sequence[str],
    partition_column: str,
    partition_value: str,
) -> None:
    assert_multicity_relation(relation)
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing columns for {relation}: {missing}")
    cursor.execute(
        f"DELETE FROM {relation} WHERE {partition_column} = %s",
        (str(partition_value),),
    )
    for rows in frame_value_batches(frame, columns):
        execute_values(
            cursor,
            f"INSERT INTO {relation} ({', '.join(columns)}) VALUES %s",
            rows,
            page_size=1000,
        )


def replace_city_metrics(
    connection,
    city: str,
    manifest: pd.DataFrame,
    buildings: pd.DataFrame,
    blocks: pd.DataFrame,
) -> None:
    """Replace all three city metric partitions in one transaction."""

    operations = [
        (SAMPLE_TABLE, manifest, _SAMPLE_COLUMNS),
        (BUILDING_TABLE, buildings, _BUILDING_COLUMNS),
        (BLOCK_TABLE, blocks, _BLOCK_COLUMNS),
    ]
    try:
        with connection.cursor() as cursor:
            for relation, frame, columns in operations:
                _replace_partition_in_cursor(
                    cursor, relation, frame, columns, "city", city
                )
        connection.commit()
    except Exception:
        connection.rollback()
        raise


def replace_city_sample_manifest(connection, city: str, frame: pd.DataFrame) -> None:
    _replace_partition(connection, SAMPLE_TABLE, frame, _SAMPLE_COLUMNS, "city", city)


def replace_city_building_metrics(connection, city: str, frame: pd.DataFrame) -> None:
    _replace_partition(connection, BUILDING_TABLE, frame, _BUILDING_COLUMNS, "city", city)


def replace_city_block_metrics(connection, city: str, frame: pd.DataFrame) -> None:
    _replace_partition(connection, BLOCK_TABLE, frame, _BLOCK_COLUMNS, "city", city)


def replace_city_retrieval_results(connection, query_city: str, frame: pd.DataFrame) -> None:
    _replace_partition(
        connection,
        RETRIEVAL_TABLE,
        frame,
        _RETRIEVAL_COLUMNS,
        "query_city",
        query_city,
    )
