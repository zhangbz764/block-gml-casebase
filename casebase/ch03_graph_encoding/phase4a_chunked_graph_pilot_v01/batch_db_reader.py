"""Read Phase 4A blocks and buildings in city-scoped, read-only batches."""

from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator

from psycopg2 import sql
from shapely import from_wkb


PHASE_DIR = Path(__file__).resolve().parent
PHASE2_DIR = PHASE_DIR.parent / "phase2_encoding_v01"
sys.path.insert(0, str(PHASE2_DIR))

from run_phase2 import METRIC_FIELDS


def iter_city_chunks(
    rows: list[dict], chunk_size: int = 100
) -> Iterator[tuple[str, str, list[dict]]]:
    """Yield stable chunks that never mix cities."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    ordered = sorted(rows, key=lambda row: (str(row["city"]), str(row["block_id"])))
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in ordered:
        grouped[str(row["city"])].append(row)
    for city in sorted(grouped):
        city_rows = grouped[city]
        for offset in range(0, len(city_rows), chunk_size):
            number = offset // chunk_size + 1
            yield city, f"chunk_{number:04d}", city_rows[offset : offset + chunk_size]


def fetch_block_batch(
    connection,
    city: str,
    block_ids: list[str],
) -> dict[str, dict[str, Any]]:
    """Fetch block metrics and metric-projected boundaries for one city."""
    selected_columns = sql.SQL(", ").join(
        sql.SQL("v.{} AS {}").format(sql.Identifier(name), sql.Identifier(name))
        for name in METRIC_FIELDS
    )
    statement = sql.SQL(
        """
        SELECT v.block_id, v.city, v.country, m.city_query,
               m.lod1_metric_srid AS metric_srid, ST_SRID(v.geom) AS storage_srid,
               {selected_columns},
               ST_AsBinary(ST_Transform(ST_Force2D(v.geom), m.lod1_metric_srid)) AS geometry_wkb_metric
        FROM summary.lod1_candidate_blocks v
        JOIN summary.city_name_map m ON m.city = v.city
        WHERE v.city = %s AND v.block_id = ANY(%s)
        ORDER BY v.block_id
        """
    ).format(selected_columns=selected_columns)
    with connection.cursor() as cursor:
        cursor.execute(statement, (city, block_ids))
        columns = [item.name for item in cursor.description]
        fetched = [dict(zip(columns, row)) for row in cursor.fetchall()]
    blocks: dict[str, dict[str, Any]] = {}
    for block in fetched:
        block["geometry"] = from_wkb(bytes(block.pop("geometry_wkb_metric")))
        blocks[str(block["block_id"])] = block
    return blocks


def fetch_building_batch(
    connection,
    city_query: str,
    metric_srid: int,
    block_ids: list[str],
) -> dict[str, list[dict]]:
    """Fetch and group all buildings for one city-scoped block batch."""
    if not re.fullmatch(r"[a-z0-9_]+", city_query):
        raise ValueError(f"Unexpected city_query: {city_query!r}")
    relation = f"{city_query}_buildings_lod1"
    statement = sql.SQL(
        """
        SELECT block_id, building_id, citygml_id, height, area, perimeter,
               ST_AsBinary(ST_Transform(ST_Force2D(geom_2d), %s)) AS geometry_wkb_metric
        FROM lod1.{}
        WHERE block_id = ANY(%s)
        ORDER BY block_id, building_id
        """
    ).format(sql.Identifier(relation))
    with connection.cursor() as cursor:
        cursor.execute(statement, (int(metric_srid), block_ids))
        columns = [item.name for item in cursor.description]
        fetched = [dict(zip(columns, row)) for row in cursor.fetchall()]
    grouped: dict[str, list[dict]] = {str(block_id): [] for block_id in block_ids}
    for building in fetched:
        block_id = str(building.pop("block_id"))
        building["geometry"] = from_wkb(bytes(building.pop("geometry_wkb_metric")))
        grouped.setdefault(block_id, []).append(building)
    return grouped


def estimate_transfer_bytes(blocks: dict[str, dict], buildings: dict[str, list[dict]]) -> int:
    """Approximate Python-side payload size, dominated by geometry WKB."""
    geometry_bytes = sum(len(block["geometry"].wkb) for block in blocks.values())
    geometry_bytes += sum(
        len(building["geometry"].wkb)
        for rows in buildings.values()
        for building in rows
    )
    return int(geometry_bytes)
