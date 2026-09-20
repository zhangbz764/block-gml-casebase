"""Collect a compact, read-only database snapshot for the phase 1 audit."""

from __future__ import annotations

import json
import argparse
import sys
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path

from psycopg2 import sql

CH03_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CH03_DIR))

from shared.db_readonly import readonly_connection  # noqa: E402


def json_default(value):
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    raise TypeError(f"Unsupported JSON type: {type(value)!r}")


def fetch_dicts(conn, sql, params=None):
    with conn.cursor() as cur:
        cur.execute(sql, params)
        columns = [item.name for item in cur.description]
        return [dict(zip(columns, row)) for row in cur.fetchall()]


def collect_snapshot(conn):
    snapshot = {}
    snapshot["database_identity"] = fetch_dicts(
        conn,
        """
        SELECT current_database() AS database_name,
               current_user AS database_user,
               current_setting('transaction_read_only') AS transaction_read_only,
               current_setting('server_version') AS server_version,
               postgis_lib_version() AS postgis_version;
        """,
    )
    snapshot["schema_table_counts"] = fetch_dicts(
        conn,
        """
        SELECT table_schema AS schema_name, COUNT(*) AS relation_count
        FROM information_schema.tables
        WHERE table_schema IN ('block', 'lod1', 'lod2', 'summary')
        GROUP BY table_schema
        ORDER BY table_schema;
        """,
    )
    snapshot["experiment_schemas"] = fetch_dicts(
        conn,
        """
        SELECT s.schema_name, COUNT(t.table_name) AS relation_count
        FROM information_schema.schemata s
        LEFT JOIN information_schema.tables t ON t.table_schema = s.schema_name
        WHERE s.schema_name LIKE 'exp\\_%' ESCAPE '\\'
        GROUP BY s.schema_name
        ORDER BY s.schema_name;
        """,
    )
    snapshot["summary_relations"] = fetch_dicts(
        conn,
        """
        SELECT table_name, table_type
        FROM information_schema.tables
        WHERE table_schema = 'summary'
        ORDER BY table_name;
        """,
    )
    snapshot["summary_population_counts"] = fetch_dicts(
        conn,
        """
        SELECT 'lod1_candidate_blocks' AS relation_name,
               COUNT(*) AS row_count, COUNT(DISTINCT city) AS city_count
        FROM summary.lod1_candidate_blocks
        UNION ALL
        SELECT 'lod1_valid_blocks', COUNT(*), COUNT(DISTINCT city)
        FROM summary.lod1_valid_blocks
        UNION ALL
        SELECT 'lod1_valid_eum_blocks', COUNT(*), COUNT(DISTINCT city)
        FROM summary.lod1_valid_eum_blocks
        ORDER BY relation_name;
        """,
    )
    snapshot["candidate_columns"] = fetch_dicts(
        conn,
        """
        SELECT ordinal_position, column_name, data_type, udt_name, is_nullable
        FROM information_schema.columns
        WHERE table_schema = 'summary' AND table_name = 'lod1_candidate_blocks'
        ORDER BY ordinal_position;
        """,
    )
    snapshot["city_map_columns"] = fetch_dicts(
        conn,
        """
        SELECT ordinal_position, column_name, data_type, udt_name, is_nullable
        FROM information_schema.columns
        WHERE table_schema = 'summary' AND table_name = 'city_name_map'
        ORDER BY ordinal_position;
        """,
    )
    snapshot["city_map"] = fetch_dicts(
        conn,
        "SELECT * FROM summary.city_name_map ORDER BY 1;",
    )
    snapshot["lod1_building_field_coverage"] = fetch_dicts(
        conn,
        """
        WITH lod1_tables AS (
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'lod1'
              AND table_name LIKE '%_buildings_lod1'
        )
        SELECT c.column_name, c.data_type, c.udt_name,
               COUNT(DISTINCT c.table_name) AS tables_with_field,
               (SELECT COUNT(*) FROM lod1_tables) AS total_building_tables
        FROM information_schema.columns c
        WHERE c.table_schema = 'lod1'
          AND c.table_name IN (SELECT table_name FROM lod1_tables)
        GROUP BY c.column_name, c.data_type, c.udt_name
        ORDER BY MIN(c.ordinal_position), c.column_name;
        """,
    )
    snapshot["lod2_building_field_coverage"] = fetch_dicts(
        conn,
        """
        WITH lod2_tables AS (
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'lod2'
              AND table_name LIKE '%_buildings_lod2'
        )
        SELECT c.column_name, c.data_type, c.udt_name,
               COUNT(DISTINCT c.table_name) AS tables_with_field,
               (SELECT COUNT(*) FROM lod2_tables) AS total_building_tables
        FROM information_schema.columns c
        WHERE c.table_schema = 'lod2'
          AND c.table_name IN (SELECT table_name FROM lod2_tables)
        GROUP BY c.column_name, c.data_type, c.udt_name
        ORDER BY MIN(c.ordinal_position), c.column_name;
        """,
    )
    snapshot["building_field_gaps"] = fetch_dicts(
        conn,
        """
        WITH target_fields(schema_name, field_name) AS (
            VALUES ('lod1', 'area'), ('lod1', 'perimeter'),
                   ('lod2', 'area'), ('lod2', 'perimeter')
        ), building_tables AS (
            SELECT table_schema, table_name
            FROM information_schema.tables
            WHERE (table_schema = 'lod1' AND table_name LIKE '%_buildings_lod1')
               OR (table_schema = 'lod2' AND table_name LIKE '%_buildings_lod2')
        )
        SELECT t.schema_name, t.field_name, b.table_name
        FROM target_fields t
        JOIN building_tables b ON b.table_schema = t.schema_name
        WHERE NOT EXISTS (
            SELECT 1 FROM information_schema.columns c
            WHERE c.table_schema = b.table_schema
              AND c.table_name = b.table_name
              AND c.column_name = t.field_name
        )
        ORDER BY t.schema_name, t.field_name, b.table_name;
        """,
    )
    snapshot["candidate_overall"] = fetch_dicts(
        conn,
        """
        SELECT COUNT(*) AS candidate_blocks,
               COUNT(DISTINCT city) AS cities,
               SUM(lod1_building_count) AS matched_building_records,
               MIN(lod1_building_count) AS min_buildings,
               percentile_disc(0.25) WITHIN GROUP (ORDER BY lod1_building_count) AS p25,
               percentile_disc(0.50) WITHIN GROUP (ORDER BY lod1_building_count) AS median,
               percentile_disc(0.75) WITHIN GROUP (ORDER BY lod1_building_count) AS p75,
               percentile_disc(0.90) WITHIN GROUP (ORDER BY lod1_building_count) AS p90,
               percentile_disc(0.95) WITHIN GROUP (ORDER BY lod1_building_count) AS p95,
               percentile_disc(0.99) WITHIN GROUP (ORDER BY lod1_building_count) AS p99,
               MAX(lod1_building_count) AS max_buildings,
               COUNT(*) FILTER (WHERE lod1_building_count = 0) AS empty_blocks,
               COUNT(*) FILTER (WHERE lod1_building_count = 1) AS single_building_blocks,
               COUNT(*) FILTER (WHERE lod1_building_count >= 500) AS blocks_ge_500,
               COUNT(*) FILTER (WHERE lod1_building_count >= 1000) AS blocks_ge_1000
        FROM summary.lod1_candidate_blocks;
        """,
    )
    snapshot["candidate_by_city"] = fetch_dicts(
        conn,
        """
        SELECT city,
               COUNT(*) AS candidate_blocks,
               SUM(lod1_building_count) AS matched_building_records,
               MIN(lod1_building_count) AS min_buildings,
               percentile_disc(0.25) WITHIN GROUP (ORDER BY lod1_building_count) AS p25,
               percentile_disc(0.50) WITHIN GROUP (ORDER BY lod1_building_count) AS median,
               percentile_disc(0.75) WITHIN GROUP (ORDER BY lod1_building_count) AS p75,
               percentile_disc(0.90) WITHIN GROUP (ORDER BY lod1_building_count) AS p90,
               percentile_disc(0.95) WITHIN GROUP (ORDER BY lod1_building_count) AS p95,
               percentile_disc(0.99) WITHIN GROUP (ORDER BY lod1_building_count) AS p99,
               MAX(lod1_building_count) AS max_buildings,
               COUNT(*) FILTER (WHERE lod1_building_count = 0) AS empty_blocks,
               COUNT(*) FILTER (WHERE lod1_building_count = 1) AS single_building_blocks,
               COUNT(*) FILTER (WHERE lod1_building_count >= 500) AS blocks_ge_500,
               COUNT(*) FILTER (WHERE lod1_building_count >= 1000) AS blocks_ge_1000
        FROM summary.lod1_candidate_blocks
        GROUP BY city
        ORDER BY city;
        """,
    )
    snapshot["relation_estimates"] = fetch_dicts(
        conn,
        """
        SELECT n.nspname AS schema_name, c.relname AS relation_name,
               c.reltuples::bigint AS estimated_rows
        FROM pg_class c
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname IN ('block', 'lod1', 'lod2')
          AND c.relkind IN ('r', 'p')
          AND (c.relname LIKE '%_blocks'
               OR c.relname LIKE '%_buildings_lod1'
               OR c.relname LIKE '%_buildings_lod2')
        ORDER BY n.nspname, c.relname;
        """,
    )
    snapshot["geometry_columns"] = fetch_dicts(
        conn,
        """
        SELECT f_table_schema AS schema_name, f_table_name AS relation_name,
               f_geometry_column AS geometry_column, coord_dimension,
               srid, type AS geometry_type
        FROM public.geometry_columns
        WHERE f_table_schema IN ('block', 'lod1', 'lod2', 'summary')
        ORDER BY f_table_schema, f_table_name, f_geometry_column;
        """,
    )
    snapshot["sample_blocks"] = fetch_dicts(
        conn,
        """
        WITH targets(city, sample_band, target_quantile) AS (
            VALUES ('Luxembourg', 'small', 0.10::double precision),
                   ('Luxembourg', 'medium', 0.50::double precision),
                   ('Luxembourg', 'large', 0.90::double precision),
                   ('Berlin', 'small', 0.10::double precision),
                   ('Berlin', 'medium', 0.50::double precision),
                   ('Berlin', 'large', 0.90::double precision),
                   ('Tokyo', 'small', 0.10::double precision),
                   ('Tokyo', 'medium', 0.50::double precision),
                   ('Tokyo', 'large', 0.90::double precision)
        ), ranked AS (
            SELECT city, block_id, lod1_building_count,
                   CUME_DIST() OVER (
                       PARTITION BY city ORDER BY lod1_building_count, block_id
                   ) AS quantile_position
            FROM summary.lod1_candidate_blocks
            WHERE city IN ('Luxembourg', 'Berlin', 'Tokyo')
        )
        SELECT t.city, t.sample_band, r.block_id, r.lod1_building_count,
               r.quantile_position,
               v.area_m2, v.bcr, v.far, v.avg_nn_distance,
               ST_SRID(v.geom) AS block_srid,
               ST_IsValid(v.geom) AS block_geom_valid,
               ST_IsEmpty(v.geom) AS block_geom_empty
        FROM targets t
        CROSS JOIN LATERAL (
            SELECT * FROM ranked r
            WHERE r.city = t.city
            ORDER BY ABS(r.quantile_position - t.target_quantile), r.block_id
            LIMIT 1
        ) r
        JOIN summary.lod1_candidate_blocks v
          ON v.city = r.city AND v.block_id = r.block_id
        ORDER BY t.city, t.target_quantile;
        """,
    )
    city_query_by_name = {
        row["city"]: row["city_query"] for row in snapshot["city_map"]
    }
    sample_trace = []
    for sample in snapshot["sample_blocks"]:
        city_query = city_query_by_name[sample["city"]]
        building_table = f"{city_query}_buildings_lod1"
        statement = sql.SQL(
            """
            SELECT COUNT(*) AS actual_building_records,
                   COUNT(DISTINCT b.building_id) AS distinct_building_ids,
                   COUNT(*) FILTER (WHERE b.building_id IS NULL) AS null_building_ids,
                   COUNT(*) FILTER (WHERE b.citygml_id IS NOT NULL) AS citygml_id_non_null,
                   COUNT(*) FILTER (WHERE b.geom_2d IS NOT NULL) AS geometry_non_null,
                   COUNT(*) FILTER (
                       WHERE b.geom_2d IS NOT NULL AND ST_IsValid(b.geom_2d)
                   ) AS geometry_valid,
                   COUNT(*) FILTER (
                       WHERE b.geom_2d IS NOT NULL AND NOT ST_IsEmpty(b.geom_2d)
                   ) AS geometry_non_empty,
                   COUNT(*) FILTER (WHERE b.height IS NOT NULL) AS height_non_null,
                   COUNT(*) FILTER (WHERE b.area IS NOT NULL) AS area_non_null,
                   COUNT(*) FILTER (WHERE b.perimeter IS NOT NULL) AS perimeter_non_null,
                   COUNT(*) FILTER (WHERE b.floor_count IS NOT NULL) AS floor_count_non_null,
                   COUNT(*) FILTER (WHERE b.function IS NOT NULL) AS function_non_null,
                   COUNT(*) FILTER (WHERE b.ground_z IS NOT NULL) AS ground_z_non_null,
                   COUNT(*) FILTER (
                       WHERE b.geom_2d IS NOT NULL
                         AND ST_IsValid(b.geom_2d)
                         AND ST_Covers(v.geom, ST_Centroid(b.geom_2d))
                   ) AS centroid_covered_by_block,
                   COUNT(*) FILTER (
                       WHERE b.geom_2d IS NOT NULL
                         AND ST_IsValid(b.geom_2d)
                         AND NOT ST_Covers(v.geom, ST_Centroid(b.geom_2d))
                   ) AS centroid_outside_block,
                   MIN(ST_SRID(b.geom_2d)) FILTER (WHERE b.geom_2d IS NOT NULL)
                       AS building_srid,
                   COUNT(*) - COUNT(DISTINCT ST_AsEWKB(b.geom_2d))
                       FILTER (WHERE b.geom_2d IS NOT NULL) AS duplicate_geometry_records
            FROM lod1.{} b
            JOIN summary.lod1_candidate_blocks v
              ON v.city = %s AND v.block_id = %s
            WHERE b.block_id = %s;
            """
        ).format(sql.Identifier(building_table))
        trace = fetch_dicts(
            conn,
            statement,
            (sample["city"], sample["block_id"], sample["block_id"]),
        )[0]
        sample_trace.append(
            {
                **sample,
                "building_relation": f"lod1.{building_table}",
                **trace,
            }
        )
    snapshot["sample_trace"] = sample_trace
    return snapshot


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--section", help="Print one snapshot section only")
    parser.add_argument("--output", type=Path, help="Optional JSON output path")
    args = parser.parse_args()
    with readonly_connection() as connection:
        data = collect_snapshot(connection)
    if args.section:
        if args.section not in data:
            raise SystemExit(f"Unknown section: {args.section}")
        data = {args.section: data[args.section]}
    rendered = json.dumps(data, ensure_ascii=False, indent=2, default=json_default)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)
