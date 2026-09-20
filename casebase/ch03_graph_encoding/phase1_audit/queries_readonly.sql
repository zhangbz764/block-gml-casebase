-- Chapter 3 phase 1 audit: representative read-only queries.
-- Connection parameters are loaded from the repository root .env by db_readonly.py.
-- No password or connection string is stored here.

BEGIN;
SET TRANSACTION READ ONLY;
SET LOCAL statement_timeout = '120s';

-- Database identity and read-only state.
SELECT current_database() AS database_name,
       current_setting('transaction_read_only') AS transaction_read_only,
       current_setting('server_version') AS server_version,
       postgis_lib_version() AS postgis_version;

-- Candidate, historical valid, and historical EUM table populations.
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

-- Overall potential graph-size distribution.
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

-- Per-city potential graph-size distribution.
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

-- Building-table field coverage; this avoids scanning building rows.
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

-- Sample traces. Only counts and quality flags are returned; geometry is not exported.
SELECT COUNT(*) AS actual_building_records,
       COUNT(DISTINCT building_id) AS distinct_building_ids,
       COUNT(*) FILTER (WHERE geom_2d IS NOT NULL) AS geometry_non_null,
       COUNT(*) FILTER (WHERE geom_2d IS NOT NULL AND ST_IsValid(geom_2d)) AS geometry_valid,
       COUNT(*) FILTER (WHERE height IS NOT NULL) AS height_non_null
FROM lod1.berlin_buildings_lod1
WHERE block_id IN ('DE_BE_002071', 'DE_BE_010880', 'DE_BE_005279');

SELECT COUNT(*) AS actual_building_records,
       COUNT(DISTINCT building_id) AS distinct_building_ids,
       COUNT(*) FILTER (WHERE geom_2d IS NOT NULL) AS geometry_non_null,
       COUNT(*) FILTER (WHERE geom_2d IS NOT NULL AND ST_IsValid(geom_2d)) AS geometry_valid,
       COUNT(*) FILTER (WHERE height IS NOT NULL) AS height_non_null
FROM lod1.luxembourg_buildings_lod1
WHERE block_id IN ('LU_LU_000408', 'LU_LU_000326', 'LU_LU_000454');

SELECT COUNT(*) AS actual_building_records,
       COUNT(DISTINCT building_id) AS distinct_building_ids,
       COUNT(*) FILTER (WHERE geom_2d IS NOT NULL) AS geometry_non_null,
       COUNT(*) FILTER (WHERE geom_2d IS NOT NULL AND ST_IsValid(geom_2d)) AS geometry_valid,
       COUNT(*) FILTER (WHERE height IS NOT NULL) AS height_non_null
FROM lod1.tokyo_buildings_lod1
WHERE block_id IN ('JP_TK_047746', 'JP_TK_054219', 'JP_TK_058267');

ROLLBACK;
