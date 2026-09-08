"""LOD1 街区与建筑指标的统一计算入口。

数据库几何继续以 EPSG:4326 存储。面积、周长使用 PostGIS geography；
最小旋转外接矩形、凸包、方向和距离等欧氏平面运算，按城市读取
``summary.city_name_map.lod1_metric_srid`` 后临时投影。

本模块只计算确定性指标，不包含 Isolation Forest、聚类或城市特例修复。
汇总函数通过 ``target_table`` 可写入独立候选表，默认值仅用于兼容旧调用。
所有函数接收已有数据库连接，不创建或关闭连接，也不会在导入时执行 SQL。
"""

from collections import Counter
import math
import re

import numpy as np
import pandas as pd
from psycopg2.extras import execute_values
from scipy.spatial import KDTree
from tqdm import tqdm


SUMMARY_TABLE = "summary.lod1_valid_blocks"
CANDIDATE_TABLE = "summary.lod1_candidate_blocks"
CITY_MAP_TABLE = "summary.city_name_map"
BASE_METRIC_BACKUP_TABLE = "summary.a_fix_r02_lod1_block_metrics_backup"
SUMMARY_BACKUP_TABLE = "summary.a_fix_r02_lod1_summary_backup"
CANDIDATE_BACKUP_TABLE = "summary.a_fix_r03_lod1_candidate_backup"
_IDENTIFIER_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_QUALIFIED_IDENTIFIER_RE = re.compile(r"^[a-z][a-z0-9_]*\.[a-z][a-z0-9_]*$")


# ---------------------------------------------------------------------------
# 纯计算函数
# ---------------------------------------------------------------------------

def normalized_elongation(side_a, side_b):
    """返回短边/长边，范围为 ``(0, 1]``；退化输入返回 ``None``。"""
    if side_a is None or side_b is None:
        return None
    side_a, side_b = float(side_a), float(side_b)
    if not math.isfinite(side_a) or not math.isfinite(side_b):
        return None
    if side_a <= 0 or side_b <= 0:
        return None
    return min(side_a, side_b) / max(side_a, side_b)


def axial_mean_deg(angles_deg, zero_tolerance=1e-12):
    """计算 180° 周期的轴向圆平均，返回 ``[0, 180)``。"""
    angles = list(angles_deg)
    if not angles:
        return None
    mean_cos2 = np.mean([math.cos(math.radians(2.0 * a)) for a in angles])
    mean_sin2 = np.mean([math.sin(math.radians(2.0 * a)) for a in angles])
    if math.hypot(mean_cos2, mean_sin2) < zero_tolerance:
        return None
    return (0.5 * math.degrees(math.atan2(mean_sin2, mean_cos2))) % 180.0


def circular_axis_difference_deg(angle_a, angle_b):
    """返回两个无向轴之间的最小夹角，范围为 ``[0, 90]``。"""
    raw = abs(float(angle_a) - float(angle_b)) % 180.0
    return min(raw, 180.0 - raw)


def orientation_entropy_deg(angles_deg, bins=18):
    """计算 ``[0, 180)`` 轴向角分箱后的 Shannon 熵。"""
    angles = list(angles_deg)
    if not angles:
        return None
    if bins <= 0:
        raise ValueError("bins 必须为正整数")
    width = 180.0 / bins
    counts = Counter(min(int((float(a) % 180.0) / width), bins - 1) for a in angles)
    total = len(angles)
    return -sum((n / total) * math.log(n / total) for n in counts.values())


def _compute_block_nn_distance(coords):
    """对单个街区内的米制质心坐标计算平均最近邻距离。"""
    if len(coords) < 2:
        return None
    distances, _ = KDTree(coords).query(coords, k=2)
    return float(np.mean(distances[:, 1]))


# ---------------------------------------------------------------------------
# 元数据与安全检查
# ---------------------------------------------------------------------------

def _validate_identifier(value, label="identifier"):
    if not isinstance(value, str) or not _IDENTIFIER_RE.fullmatch(value):
        raise ValueError(f"非法 {label}: {value!r}")
    return value


def _validate_qualified_identifier(value, label="qualified identifier"):
    if not isinstance(value, str) or not _QUALIFIED_IDENTIFIER_RE.fullmatch(value):
        raise ValueError(f"非法 {label}: {value!r}")
    return value


def _relation_exists(conn, relation_name):
    with conn.cursor() as cur:
        cur.execute("SELECT to_regclass(%s);", (relation_name,))
        return cur.fetchone()[0] is not None


def _require_metric_table(conn, target_table):
    target_table = _validate_qualified_identifier(target_table, "target_table")
    if not _relation_exists(conn, target_table):
        raise RuntimeError(f"指标表不存在：{target_table}")
    return target_table


def _safe_rollback(conn):
    """仅在连接仍可用时回滚；返回是否实际执行了回滚。"""
    if conn is None or getattr(conn, "closed", 1):
        return False
    try:
        conn.rollback()
        return True
    except Exception:
        return False


def get_lod1_metric_plan(conn, cities=None, city_map_table=CITY_MAP_TABLE):
    """读取并验证城市显示名、源 SRID 和测度 SRID。"""
    if cities is not None:
        cities = [_validate_identifier(c, "city_query") for c in cities]
    with conn.cursor() as cur:
        if cities is None:
            cur.execute(f"""
                SELECT city_query, city, lod1_source_srid, lod1_metric_srid
                FROM {city_map_table}
                WHERE lod1_metric_srid IS NOT NULL
                ORDER BY city_query;
            """)
        else:
            cur.execute(f"""
                SELECT city_query, city, lod1_source_srid, lod1_metric_srid
                FROM {city_map_table}
                WHERE city_query = ANY(%s);
            """, (cities,))
        rows = cur.fetchall()

    by_city = {
        row[0]: {
            "city_code": row[0], "display_city": row[1],
            "source_srid": row[2], "metric_srid": row[3],
        }
        for row in rows
    }
    ordered_codes = cities if cities is not None else sorted(by_city)
    missing = [city for city in ordered_codes if city not in by_city]
    if missing:
        raise RuntimeError("city_name_map 缺少 LOD1 城市：" + ", ".join(missing))
    null_metric = [city for city in ordered_codes if by_city[city]["metric_srid"] is None]
    if null_metric:
        raise RuntimeError("以下城市未设置 lod1_metric_srid：" + ", ".join(null_metric))

    metric_srids = sorted({int(by_city[c]["metric_srid"]) for c in ordered_codes})
    with conn.cursor() as cur:
        cur.execute("""
            SELECT srid, srtext FROM spatial_ref_sys WHERE srid = ANY(%s);
        """, (metric_srids,))
        srs_by_id = {int(srid): srtext for srid, srtext in cur.fetchall()}
    missing_srids = sorted(set(metric_srids) - set(srs_by_id))
    if missing_srids:
        raise RuntimeError("PostGIS 无法识别测度 EPSG：" + ", ".join(map(str, missing_srids)))
    nonprojected = [
        srid for srid in metric_srids
        if "PROJCRS[" not in srs_by_id[srid].upper()
        and "PROJCS[" not in srs_by_id[srid].upper()
    ]
    if nonprojected:
        raise RuntimeError("lod1_metric_srid 必须为投影 CRS：" + ", ".join(map(str, nonprojected)))
    return [by_city[city] for city in ordered_codes]


def validate_lod1_metric_environment(conn, cities=None, require_summary=False):
    """只读预检表、4326 存储 SRID 和测度 CRS。"""
    plan = get_lod1_metric_plan(conn, cities)
    problems = []
    if require_summary and not _relation_exists(conn, SUMMARY_TABLE):
        problems.append(f"缺少 {SUMMARY_TABLE}")
    for item in plan:
        city = item["city_code"]
        block_table = f"block.{city}_blocks"
        building_table = f"lod1.{city}_buildings_lod1"
        missing_relation = False
        for relation in (block_table, building_table):
            if not _relation_exists(conn, relation):
                problems.append(f"缺少 {relation}")
                missing_relation = True
        if missing_relation:
            continue
        with conn.cursor() as cur:
            cur.execute(f"""
                SELECT ST_SRID(geom), COUNT(*) FROM {block_table}
                WHERE geom IS NOT NULL GROUP BY ST_SRID(geom);
            """)
            block_srids = cur.fetchall()
            cur.execute(f"""
                SELECT ST_SRID(geom_2d), COUNT(*) FROM {building_table}
                WHERE geom_2d IS NOT NULL GROUP BY ST_SRID(geom_2d);
            """)
            building_srids = cur.fetchall()
        if not block_srids or any(row[0] != 4326 for row in block_srids):
            problems.append(f"{block_table} 的 SRID 分布异常：{block_srids}")
        if not building_srids or any(row[0] != 4326 for row in building_srids):
            problems.append(f"{building_table} 的 SRID 分布异常：{building_srids}")
    if problems:
        raise RuntimeError("LOD1 指标环境预检未通过：\n- " + "\n- ".join(problems))
    return plan


# ---------------------------------------------------------------------------
# SQL 构造：平面几何部分全部使用城市 metric_srid
# ---------------------------------------------------------------------------

def build_lod1_block_geometry_update_sql(city):
    """构造街区 area/compactness/elongation 更新 SQL。"""
    city = _validate_identifier(city, "city_query")
    table = f"block.{city}_blocks"
    return f"""
        WITH prepared AS (
            SELECT b.block_id, b.geom, ST_IsValid(b.geom) AS is_valid,
                CASE WHEN ST_IsValid(b.geom) AND NOT ST_IsEmpty(b.geom)
                     THEN ST_Transform(ST_Force2D(b.geom), %(metric_srid)s) END AS geom_m
            FROM {table} b WHERE b.geom IS NOT NULL
        ), envelopes AS (
            SELECT block_id, geom, is_valid, ST_OrientedEnvelope(geom_m) AS mbr
            FROM prepared
        ), dimensions AS (
            SELECT block_id, geom, is_valid,
                ST_Distance(ST_PointN(ST_ExteriorRing(mbr), 1),
                            ST_PointN(ST_ExteriorRing(mbr), 2)) AS edge_1,
                ST_Distance(ST_PointN(ST_ExteriorRing(mbr), 2),
                            ST_PointN(ST_ExteriorRing(mbr), 3)) AS edge_2
            FROM envelopes
        )
        UPDATE {table} b SET
            geom_is_valid = d.is_valid,
            area_m2 = CASE WHEN d.is_valid THEN ST_Area(d.geom::geography) END,
            compactness = CASE
                WHEN d.is_valid AND ST_Perimeter(d.geom::geography) > 0
                THEN 4.0 * PI() * ST_Area(d.geom::geography)
                     / POWER(ST_Perimeter(d.geom::geography), 2) END,
            elongation = CASE
                WHEN d.is_valid AND d.edge_1 > 0 AND d.edge_2 > 0
                THEN LEAST(edge_1, edge_2)
                     / NULLIF(GREATEST(edge_1, edge_2), 0) END
        FROM dimensions d WHERE b.block_id = d.block_id;
    """


def build_lod1_block_building_update_sql(city):
    city = _validate_identifier(city, "city_query")
    block_table = f"block.{city}_blocks"
    building_table = f"lod1.{city}_buildings_lod1"
    return f"""
        WITH aggregated AS (
            SELECT b.block_id, COUNT(*) AS building_count,
                   SUM(CASE
                       WHEN b.geom_2d IS NOT NULL
                        AND ST_IsValid(b.geom_2d)
                        AND NOT ST_IsEmpty(b.geom_2d)
                       THEN ST_Area(b.geom_2d::geography)
                   END) AS total_footprint
            FROM {building_table} b
            WHERE b.block_id IS NOT NULL
            GROUP BY b.block_id
        )
        UPDATE {block_table} blk SET
            lod1_building_count = a.building_count,
            lod1_bcr = CASE WHEN blk.area_m2 > 0
                            THEN a.total_footprint / blk.area_m2 END
        FROM aggregated a WHERE blk.block_id = a.block_id;
    """


def build_lod1_summary_block_shape_update_sql(city, target_table=SUMMARY_TABLE):
    _validate_identifier(city, "city_query")
    target_table = _validate_qualified_identifier(target_table, "target_table")
    return f"""
        WITH projected AS (
            SELECT block_id,
                   ST_Transform(ST_Force2D(geom), %(metric_srid)s) AS geom_m
            FROM {target_table}
            WHERE city = %(display_city)s AND geom IS NOT NULL
              AND ST_IsValid(geom) AND NOT ST_IsEmpty(geom)
        ), measured AS (
            SELECT block_id, ST_Area(geom_m) AS area_m2,
                   ST_Area(ST_ConvexHull(geom_m)) AS hull_area_m2,
                   ST_Area(ST_OrientedEnvelope(geom_m)) AS envelope_area_m2
            FROM projected
        )
        UPDATE {target_table} v SET
            convexity = m.area_m2 / NULLIF(m.hull_area_m2, 0),
            rectangularity = m.area_m2 / NULLIF(m.envelope_area_m2, 0)
        FROM measured m
        WHERE v.block_id = m.block_id AND v.city = %(display_city)s;
    """


def build_lod1_building_shape_update_sql(city, target_table=SUMMARY_TABLE):
    city = _validate_identifier(city, "city_query")
    target_table = _validate_qualified_identifier(target_table, "target_table")
    building_table = f"lod1.{city}_buildings_lod1"
    return f"""
        WITH projected AS (
            SELECT b.block_id,
                   ST_Transform(ST_Force2D(b.geom_2d), %(metric_srid)s) AS geom_m
            FROM {building_table} b
            JOIN {target_table} v ON v.block_id = b.block_id
                                  AND v.city = %(display_city)s
            WHERE b.block_id IS NOT NULL AND b.geom_2d IS NOT NULL
              AND ST_IsValid(b.geom_2d) AND NOT ST_IsEmpty(b.geom_2d)
        ), measured AS (
            SELECT block_id, ST_Area(geom_m) AS area_m2,
                   ST_Perimeter(geom_m) AS perimeter_m,
                   ST_Area(ST_OrientedEnvelope(geom_m)) AS envelope_area_m2,
                   ST_Area(ST_ConvexHull(geom_m)) AS hull_area_m2
            FROM projected WHERE ST_Area(geom_m) > 0
        ), aggregated AS (
            SELECT block_id,
                AVG(CASE WHEN area_m2 >= %(min_footprint_area_m2)s
                         THEN POWER(perimeter_m, 2) / (4.0 * PI() * area_m2) END)
                    AS avg_complexity,
                AVG(CASE WHEN area_m2 >= %(min_footprint_area_m2)s
                         THEN LEAST(1.0,
                             area_m2 / NULLIF(envelope_area_m2, 0)) END)
                    AS avg_rectangularity,
                AVG(CASE WHEN area_m2 >= %(min_footprint_area_m2)s
                         THEN LEAST(1.0,
                             area_m2 / NULLIF(hull_area_m2, 0)) END)
                    AS avg_convexity
            FROM measured GROUP BY block_id
        )
        UPDATE {target_table} v SET
            avg_complexity = a.avg_complexity,
            avg_rectangularity = a.avg_rectangularity,
            avg_convexity = a.avg_convexity
        FROM aggregated a
        WHERE v.block_id = a.block_id AND v.city = %(display_city)s;
    """


def build_lod1_orientation_update_sql(city, target_table=SUMMARY_TABLE):
    city = _validate_identifier(city, "city_query")
    target_table = _validate_qualified_identifier(target_table, "target_table")
    building_table = f"lod1.{city}_buildings_lod1"
    return f"""
        WITH projected AS (
            SELECT b.block_id,
                   ST_Transform(ST_Force2D(b.geom_2d), %(metric_srid)s) AS geom_m
            FROM {building_table} b
            JOIN {target_table} v ON v.block_id = b.block_id
                                  AND v.city = %(display_city)s
            WHERE b.block_id IS NOT NULL AND b.geom_2d IS NOT NULL
              AND ST_IsValid(b.geom_2d) AND NOT ST_IsEmpty(b.geom_2d)
        ), edge_points AS (
            SELECT block_id,
                ST_PointN(ST_ExteriorRing(ST_OrientedEnvelope(geom_m)), 1) AS p1,
                ST_PointN(ST_ExteriorRing(ST_OrientedEnvelope(geom_m)), 2) AS p2,
                ST_PointN(ST_ExteriorRing(ST_OrientedEnvelope(geom_m)), 3) AS p3
            FROM projected
            WHERE ST_Area(geom_m) > 0
              AND GeometryType(ST_OrientedEnvelope(geom_m)) = 'POLYGON'
        ), long_edges AS (
            SELECT block_id,
                CASE WHEN ST_Distance(p1,p2) >= ST_Distance(p2,p3)
                     THEN p1 ELSE p2 END AS p_start,
                CASE WHEN ST_Distance(p1,p2) >= ST_Distance(p2,p3)
                     THEN p2 ELSE p3 END AS p_end
            FROM edge_points WHERE p1 IS NOT NULL AND p2 IS NOT NULL AND p3 IS NOT NULL
        ), angles AS (
            SELECT block_id,
                MOD(DEGREES(ST_Azimuth(
                    ST_Transform(p_start,4326)::geography,
                    ST_Transform(p_end,4326)::geography
                ))::numeric, 180.0)::double precision AS angle_deg
            FROM long_edges WHERE NOT ST_Equals(p_start,p_end)
        ), trig_stats AS (
            SELECT block_id,
                AVG(COS(RADIANS(2.0 * angle_deg))) AS mean_cos2,
                AVG(SIN(RADIANS(2.0 * angle_deg))) AS mean_sin2
            FROM angles WHERE angle_deg IS NOT NULL GROUP BY block_id
        ), angle_summary AS (
            SELECT block_id,
                CASE WHEN SQRT(POWER(mean_cos2,2)+POWER(mean_sin2,2)) < 1e-12
                     THEN NULL ELSE MOD(
                         (0.5 * DEGREES(ATAN2(mean_sin2, mean_cos2)) + 180.0)::numeric,
                         180.0)::double precision END AS avg_orientation
            FROM trig_stats
        ), bin_counts AS (
            SELECT block_id, FLOOR(angle_deg/10.0)::integer AS bin_id,
                   COUNT(*)::double precision AS bin_count
            FROM angles WHERE angle_deg IS NOT NULL
            GROUP BY block_id, FLOOR(angle_deg/10.0)::integer
        ), entropy_summary AS (
            SELECT block_id, -SUM(p*LN(p)) AS orientation_entropy
            FROM (
                SELECT block_id,
                       bin_count/SUM(bin_count) OVER (PARTITION BY block_id) AS p
                FROM bin_counts
            ) probabilities GROUP BY block_id
        )
        UPDATE {target_table} v SET
            avg_orientation = a.avg_orientation,
            orientation_entropy = e.orientation_entropy
        FROM angle_summary a JOIN entropy_summary e USING (block_id)
        WHERE v.block_id = a.block_id AND v.city = %(display_city)s;
    """


def build_lod1_nn_centroid_query(city, target_table=SUMMARY_TABLE):
    city = _validate_identifier(city, "city_query")
    target_table = _validate_qualified_identifier(target_table, "target_table")
    building_table = f"lod1.{city}_buildings_lod1"
    return f"""
        SELECT b.building_id, b.block_id,
            ST_X(ST_Transform(ST_Centroid(ST_Force2D(b.geom_2d)),
                              %(metric_srid)s)) AS cx,
            ST_Y(ST_Transform(ST_Centroid(ST_Force2D(b.geom_2d)),
                              %(metric_srid)s)) AS cy
        FROM {building_table} b
        JOIN {target_table} v ON v.block_id=b.block_id
                              AND v.city=%(display_city)s
        WHERE b.block_id IS NOT NULL AND b.geom_2d IS NOT NULL
          AND ST_IsValid(b.geom_2d) AND NOT ST_IsEmpty(b.geom_2d);
    """


def build_lod1_candidate_insert_sql(city, target_table=CANDIDATE_TABLE):
    """构造候选表导入 SQL；唯一纳入条件是至少一个匹配的 LOD1 建筑对象。"""
    city = _validate_identifier(city, "city_query")
    target_table = _validate_qualified_identifier(target_table, "target_table")
    return f"""
        INSERT INTO {target_table} (
            city, country, block_id, geom, area_m2,
            elongation, compactness, lod1_building_count, bcr
        )
        SELECT city, country, block_id, geom, area_m2,
               elongation, compactness, lod1_building_count, lod1_bcr
        FROM block.{city}_blocks
        WHERE lod1_building_count >= 1;
    """


def build_lod1_density_height_volume_update_sql(city, target_table=SUMMARY_TABLE):
    """构造密度、高度、体量和底面积统计更新 SQL。"""
    city = _validate_identifier(city, "city_query")
    target_table = _validate_qualified_identifier(target_table, "target_table")
    building_table = f"lod1.{city}_buildings_lod1"
    return f"""
        WITH aggregated AS (
            SELECT b.block_id, COUNT(*) AS building_count,
                SUM(b.area) AS total_footprint,
                AVG(b.area) AS avg_footprint,
                STDDEV(b.area) AS stddev_footprint,
                AVG(b.height) FILTER (WHERE b.height>0) AS avg_height,
                SUM(b.area*b.height) FILTER (WHERE b.height>0 AND b.area>0)
                    / NULLIF(SUM(b.area) FILTER (WHERE b.height>0 AND b.area>0),0)
                    AS weighted_avg_height,
                STDDEV(b.height) FILTER (WHERE b.height>0) AS stddev_height,
                MAX(b.height) FILTER (WHERE b.height>0) AS max_height,
                MIN(b.height) FILTER (WHERE b.height>0) AS min_height,
                SUM(b.area*b.floor_count) FILTER (WHERE b.floor_count>0 AND b.area>0)
                    AS total_floor_area,
                SUM(b.area*b.height) FILTER (WHERE b.height>0 AND b.area>0)
                    AS total_volume,
                AVG(b.area*b.height) FILTER (WHERE b.height>0 AND b.area>0)
                    AS avg_volume
            FROM {building_table} b
            JOIN {target_table} v ON v.block_id=b.block_id AND v.city=%s
            WHERE b.block_id IS NOT NULL
            GROUP BY b.block_id
        )
        UPDATE {target_table} v SET
            bcr=a.total_footprint/NULLIF(v.area_m2,0),
            far=a.total_floor_area/NULLIF(v.area_m2,0),
            bcd=a.building_count/NULLIF(v.area_m2/10000.0,0),
            avh=a.avg_height, avh_w=a.weighted_avg_height,
            hsd=a.stddev_height,
            hcv=a.stddev_height/NULLIF(a.avg_height,0),
            h_max=a.max_height, h_range=a.max_height-a.min_height,
            total_volume=a.total_volume,
            avg_volume=a.avg_volume,
            avg_footprint=a.avg_footprint,
            footprint_cv=a.stddev_footprint/NULLIF(a.avg_footprint,0)
        FROM aggregated a
        WHERE v.block_id=a.block_id AND v.city=%s;
    """


def build_lod1_building_measurement_update_sql(city, mode="missing"):
    """构造建筑面积和周长更新 SQL。

    ``missing`` 只补空值；``refresh`` 重算所有有效几何，但也只执行一次 UPDATE，
    不再先把整表字段清空。
    """
    city = _validate_identifier(city, "city_query")
    if mode not in {"missing", "refresh"}:
        raise ValueError("mode 必须为 'missing' 或 'refresh'")
    table = f"lod1.{city}_buildings_lod1"
    if mode == "missing":
        return f"""
            UPDATE {table}
            SET area = CASE
                    WHEN area IS NULL THEN ST_Area(geom_2d::geography)
                    ELSE area
                END,
                perimeter = CASE
                    WHEN perimeter IS NULL THEN ST_Perimeter(geom_2d::geography)
                    ELSE perimeter
                END
            WHERE geom_2d IS NOT NULL
              AND ST_IsValid(geom_2d)
              AND NOT ST_IsEmpty(geom_2d)
              AND (area IS NULL OR perimeter IS NULL);
        """
    return f"""
        UPDATE {table}
        SET area = ST_Area(geom_2d::geography),
            perimeter = ST_Perimeter(geom_2d::geography)
        WHERE geom_2d IS NOT NULL
          AND ST_IsValid(geom_2d)
          AND NOT ST_IsEmpty(geom_2d);
    """


# ---------------------------------------------------------------------------
# 筛选前：对全部候选街区重算基础指标
# ---------------------------------------------------------------------------

def compute_lod1_source_block_metrics(
    conn, cities=None, backup_table=BASE_METRIC_BACKUP_TABLE, verbose=True
):
    """重算分城市 block 表的 LOD1 基础指标，不作筛选。

    作用对象是各城市完整的 ``block.<city>_blocks``，不是当前 summary 子集。
    完成本函数后，需要重新运行 Isolation Forest 才能得到新版筛选结果。
    """
    plan = validate_lod1_metric_environment(conn, cities, require_summary=False)
    backup_table = _validate_qualified_identifier(backup_table, "backup_table")
    with conn.cursor() as cur:
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {backup_table} (
                city_query VARCHAR NOT NULL,
                block_id VARCHAR NOT NULL,
                area_m2 NUMERIC,
                elongation NUMERIC,
                compactness NUMERIC,
                lod1_building_count INTEGER,
                lod1_bcr NUMERIC,
                geom_is_valid BOOLEAN,
                backed_up_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                PRIMARY KEY (city_query, block_id)
            );
        """)
    conn.commit()
    results = []
    for item in plan:
        city = item["city_code"]
        table = f"block.{city}_blocks"
        try:
            with conn.cursor() as cur:
                cur.execute(f"""
                    ALTER TABLE {table}
                        ADD COLUMN IF NOT EXISTS elongation NUMERIC,
                        ADD COLUMN IF NOT EXISTS compactness NUMERIC,
                        ADD COLUMN IF NOT EXISTS lod1_building_count INTEGER,
                        ADD COLUMN IF NOT EXISTS lod1_bcr NUMERIC,
                        ADD COLUMN IF NOT EXISTS geom_is_valid BOOLEAN;
                """)
                # 首次值保留；同一城市重复运行不会覆盖原始备份。
                cur.execute(f"""
                    INSERT INTO {backup_table} (
                        city_query, block_id, area_m2, elongation, compactness,
                        lod1_building_count, lod1_bcr, geom_is_valid
                    )
                    SELECT %s, block_id, area_m2, elongation, compactness,
                           lod1_building_count, lod1_bcr, geom_is_valid
                    FROM {table}
                    ON CONFLICT (city_query, block_id) DO NOTHING;
                """, (city,))
                cur.execute(
                    build_lod1_block_geometry_update_sql(city),
                    {"metric_srid": item["metric_srid"]},
                )
                geometry_rows = cur.rowcount
                # 先重置，避免建筑删除或 block_id 调整后遗留旧聚合值。
                cur.execute(f"""
                    UPDATE {table}
                    SET lod1_building_count = 0, lod1_bcr = NULL;
                """)
                cur.execute(build_lod1_block_building_update_sql(city))
                building_rows = cur.rowcount
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        results.append({
            **item,
            "geometry_rows": geometry_rows,
            "building_rows": building_rows,
        })
        if verbose:
            print(
                f"[{city}] 几何 {geometry_rows:,}，含建筑街区 {building_rows:,}，"
                f"EPSG:{item['metric_srid']}"
            )
    return results


# 兼容此前 notebook 的旧函数名；新流程使用 compute_lod1_source_block_metrics。
compute_lod1_block_screening_metrics = compute_lod1_source_block_metrics


# ---------------------------------------------------------------------------
# 汇总表创建、重建与非投影敏感指标
# ---------------------------------------------------------------------------

def build_lod1_metric_table_ddl(target_table=SUMMARY_TABLE):
    """构造 LOD1 指标汇总表 DDL。"""
    target_table = _validate_qualified_identifier(target_table, "target_table")
    _, table_name = target_table.split(".")
    index_prefix = _validate_identifier(table_name, "table_name")
    return f"""
            CREATE TABLE IF NOT EXISTS {target_table} (
                block_id VARCHAR NOT NULL PRIMARY KEY,
                city VARCHAR NOT NULL,
                country VARCHAR NOT NULL,
                geom GEOMETRY(Polygon, 4326),
                area_m2 NUMERIC,
                elongation NUMERIC,
                compactness NUMERIC,
                lod1_building_count INTEGER,
                bcr NUMERIC,
                convexity NUMERIC,
                rectangularity NUMERIC,
                far NUMERIC,
                bcd NUMERIC,
                osr NUMERIC,
                avh NUMERIC,
                avh_w NUMERIC,
                hsd NUMERIC,
                hcv NUMERIC,
                h_max NUMERIC,
                h_range NUMERIC,
                total_volume NUMERIC,
                avg_volume NUMERIC,
                avg_footprint NUMERIC,
                footprint_cv NUMERIC,
                avg_complexity NUMERIC,
                avg_rectangularity NUMERIC,
                avg_convexity NUMERIC,
                avg_orientation NUMERIC,
                orientation_entropy NUMERIC,
                avg_nn_distance NUMERIC
            );
            CREATE INDEX IF NOT EXISTS {index_prefix}_city_block_idx
                ON {target_table} (city, block_id);
            CREATE INDEX IF NOT EXISTS {index_prefix}_geom_idx
                ON {target_table} USING GIST (geom);
    """


def ensure_lod1_metric_table(conn, target_table=SUMMARY_TABLE):
    """创建指定 LOD1 指标表及索引（已存在时不覆盖）。"""
    with conn.cursor() as cur:
        cur.execute(build_lod1_metric_table_ddl(target_table))
    conn.commit()


def ensure_lod1_summary_table(conn):
    """兼容旧 notebook：创建 ``summary.lod1_valid_blocks``。"""
    return ensure_lod1_metric_table(conn, SUMMARY_TABLE)


def rebuild_lod1_summary(
    conn, cities=None, confirm=False,
    backup_table=SUMMARY_BACKUP_TABLE, verbose=True
):
    """按最新 ``is_valid_lod1=true`` 原子化重建汇总表。"""
    if not confirm:
        raise RuntimeError("重建会清空当前汇总表；确认后请显式传入 confirm=True")
    plan = validate_lod1_metric_environment(conn, cities, require_summary=True)
    backup_table = _validate_qualified_identifier(backup_table, "backup_table")
    try:
        with conn.cursor() as cur:
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {backup_table}
                    (LIKE {SUMMARY_TABLE} INCLUDING ALL);
            """)
            cur.execute(f"""
                INSERT INTO {backup_table}
                SELECT * FROM {SUMMARY_TABLE}
                ON CONFLICT (block_id) DO NOTHING;
            """)
            cur.execute(f"TRUNCATE TABLE {SUMMARY_TABLE};")
            total = 0
            for item in plan:
                city = item["city_code"]
                cur.execute(f"""
                    INSERT INTO {SUMMARY_TABLE} (
                        city, country, block_id, geom, area_m2,
                        elongation, compactness, lod1_building_count, bcr
                    )
                    SELECT city, country, block_id, geom, area_m2,
                           elongation, compactness, lod1_building_count, lod1_bcr
                    FROM block.{city}_blocks
                    WHERE is_valid_lod1 IS TRUE;
                """)
                total += cur.rowcount
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    if verbose:
        print(f"汇总表重建完成：{total:,} 个街区")
    return total


def rebuild_lod1_candidate_table(
    conn,
    cities=None,
    target_table=CANDIDATE_TABLE,
    confirm=False,
    backup_table=CANDIDATE_BACKUP_TABLE,
    verbose=True,
):
    """从分城市 block 表重建候选汇总表。

    唯一纳入条件是 ``lod1_building_count >= 1``。不根据 BCR、几何有效性、
    指标空值、旧 ``is_valid_lod1`` 或异常模型结果排除记录。
    """
    if not confirm:
        raise RuntimeError("重建会清空候选表；确认后请显式传入 confirm=True")
    target_table = _validate_qualified_identifier(target_table, "target_table")
    backup_table = _validate_qualified_identifier(backup_table, "backup_table")
    if target_table == SUMMARY_TABLE:
        raise ValueError("候选表不得使用 summary.lod1_valid_blocks")
    plan = validate_lod1_metric_environment(conn, cities, require_summary=False)
    if not _relation_exists(conn, target_table):
        raise RuntimeError(f"候选表不存在：{target_table}；请先调用 ensure_lod1_metric_table")

    try:
        with conn.cursor() as cur:
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {backup_table}
                    (LIKE {target_table} INCLUDING ALL);
            """)
            cur.execute(f"""
                INSERT INTO {backup_table}
                SELECT * FROM {target_table}
                ON CONFLICT (block_id) DO NOTHING;
            """)
            cur.execute(f"TRUNCATE TABLE {target_table};")
            city_counts = {}
            for item in plan:
                city = item["city_code"]
                cur.execute(build_lod1_candidate_insert_sql(city, target_table))
                city_counts[city] = cur.rowcount
        conn.commit()
    except Exception:
        conn.rollback()
        raise

    total = sum(city_counts.values())
    if verbose:
        print(f"候选表重建完成：{total:,} 个至少含一栋建筑的街区")
        for city, count in city_counts.items():
            print(f"  [{city}] {count:,}")
    return {"target_table": target_table, "total": total, "city_counts": city_counts}


def prepare_lod1_building_measurements(
    conn,
    cities=None,
    refresh_area_perimeter=None,
    fill_missing_floor_count=True,
    verbose=True,
    area_perimeter_mode="missing",
):
    """准备建筑底面积、周长和缺失层数；完整城市自动跳过。

    ``area_perimeter_mode='missing'`` 只补空值，不会替换非空的源字段或 0
    占位值，因此只适合已经核验过存量测度的增量流程。``'refresh'`` 才会按
    geography 口径全量重算有效几何；``'skip'`` 完全跳过面积/周长。
    旧参数 ``refresh_area_perimeter`` 仅用于兼容历史 notebook。
    """
    if refresh_area_perimeter is not None:
        area_perimeter_mode = "refresh" if refresh_area_perimeter else "skip"
    if area_perimeter_mode not in {"missing", "refresh", "skip"}:
        raise ValueError("area_perimeter_mode 必须为 'missing'、'refresh' 或 'skip'")

    plan = validate_lod1_metric_environment(conn, cities, require_summary=False)
    results = []
    for item in plan:
        city = item["city_code"]
        table = f"lod1.{city}_buildings_lod1"
        try:
            with conn.cursor() as cur:
                cur.execute(f"""
                    ALTER TABLE {table}
                        ADD COLUMN IF NOT EXISTS area DOUBLE PRECISION,
                        ADD COLUMN IF NOT EXISTS perimeter DOUBLE PRECISION;
                """)
                cur.execute(f"""
                    SELECT
                        COUNT(*) FILTER (
                            WHERE geom_2d IS NOT NULL
                              AND ST_IsValid(geom_2d)
                              AND NOT ST_IsEmpty(geom_2d)
                        ) AS valid_geom,
                        COUNT(*) FILTER (
                            WHERE geom_2d IS NOT NULL
                              AND ST_IsValid(geom_2d)
                              AND NOT ST_IsEmpty(geom_2d)
                              AND area IS NULL
                        ) AS missing_area,
                        COUNT(*) FILTER (
                            WHERE geom_2d IS NOT NULL
                              AND ST_IsValid(geom_2d)
                              AND NOT ST_IsEmpty(geom_2d)
                              AND perimeter IS NULL
                        ) AS missing_perimeter,
                        COUNT(*) FILTER (
                            WHERE (floor_count IS NULL OR floor_count <= 0)
                              AND height IS NOT NULL AND height > 0
                        ) AS missing_floor
                    FROM {table};
                """)
                valid_geom, missing_area, missing_perimeter, missing_floor = cur.fetchone()
                area_rows = 0
                floor_rows = 0
                needs_area_work = (
                    area_perimeter_mode == "refresh"
                    or (
                        area_perimeter_mode == "missing"
                        and (missing_area > 0 or missing_perimeter > 0)
                    )
                )
                if needs_area_work:
                    cur.execute(
                        build_lod1_building_measurement_update_sql(
                            city, mode=area_perimeter_mode
                        )
                    )
                    area_rows = cur.rowcount
                if fill_missing_floor_count and missing_floor > 0:
                    cur.execute(f"""
                        UPDATE {table}
                        SET floor_count = GREATEST(ROUND(height / 3.0)::integer, 1)
                        WHERE (floor_count IS NULL OR floor_count <= 0)
                          AND height IS NOT NULL AND height > 0;
                    """)
                    floor_rows = cur.rowcount
            conn.commit()
        except Exception as exc:
            rolled_back = _safe_rollback(conn)
            connection_state = "可回滚" if rolled_back else "连接已关闭或回滚不可用"
            raise RuntimeError(
                f"[{city}] 建筑通用测度失败（{connection_state}）；"
                "已完成并提交的此前城市无需重算。"
            ) from exc
        skipped = area_rows == 0 and floor_rows == 0
        results.append({
            **item,
            "measurement_rows": area_rows,
            "floor_rows": floor_rows,
            "missing_area_before": int(missing_area),
            "missing_perimeter_before": int(missing_perimeter),
            "missing_floor_before": int(missing_floor),
            "skipped": skipped,
        })
        if verbose:
            if skipped:
                print(f"[{city}] 已完整，跳过")
            else:
                print(
                    f"[{city}] 建筑面积/周长补充 {area_rows:,}，"
                    f"补层数 {floor_rows:,}"
                )
    return results


def compute_lod1_density_height_volume_metrics(
    conn, cities=None, target_table=SUMMARY_TABLE, verbose=True
):
    """计算 BCR/FAR/BCD/OSR、高度、体量和底面积统计。"""
    target_table = _require_metric_table(conn, target_table)
    plan = get_lod1_metric_plan(conn, cities)
    for item in plan:
        city = item["city_code"]
        display_city = item["display_city"]
        try:
            with conn.cursor() as cur:
                cur.execute(f"""
                    UPDATE {target_table}
                    SET bcr=NULL, far=NULL, bcd=NULL, osr=NULL,
                        avh=NULL, avh_w=NULL, hsd=NULL, hcv=NULL,
                        h_max=NULL, h_range=NULL, total_volume=NULL,
                        avg_volume=NULL, avg_footprint=NULL, footprint_cv=NULL
                    WHERE city=%s;
                """, (display_city,))
                cur.execute(
                    build_lod1_density_height_volume_update_sql(
                        city, target_table=target_table
                    ),
                    (display_city, display_city),
                )
                updated = cur.rowcount
                cur.execute(f"""
                    UPDATE {target_table}
                    SET osr=(1.0-bcr)/NULLIF(far,0)
                    WHERE city=%s AND bcr IS NOT NULL AND far>0;
                """, (display_city,))
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        if verbose:
            print(f"[{city}] 密度/高度/体量：{updated:,} 个街区")


# ---------------------------------------------------------------------------
# 汇总表中的投影敏感指标
# ---------------------------------------------------------------------------

def compute_lod1_shape_metrics(
    conn, cities=None, target_table=SUMMARY_TABLE,
    min_footprint_area_m2=1.0, verbose=True
):
    """重算街区/建筑凸度、矩形度与建筑复杂度。"""
    target_table = _require_metric_table(conn, target_table)
    plan = validate_lod1_metric_environment(conn, cities, require_summary=False)
    for item in plan:
        params = {
            "display_city": item["display_city"],
            "metric_srid": item["metric_srid"],
            "min_footprint_area_m2": float(min_footprint_area_m2),
        }
        try:
            with conn.cursor() as cur:
                cur.execute(f"""
                    UPDATE {target_table}
                    SET convexity=NULL, rectangularity=NULL,
                        avg_complexity=NULL, avg_rectangularity=NULL,
                        avg_convexity=NULL
                    WHERE city=%(display_city)s;
                """, params)
                cur.execute(
                    build_lod1_summary_block_shape_update_sql(
                        item["city_code"], target_table=target_table
                    ),
                    params,
                )
                block_rows = cur.rowcount
                cur.execute(
                    build_lod1_building_shape_update_sql(
                        item["city_code"], target_table=target_table
                    ),
                    params,
                )
                building_rows = cur.rowcount
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        if verbose:
            print(
                f"[{item['city_code']}] 形态：街区 {block_rows:,}，"
                f"建筑聚合 {building_rows:,}，EPSG:{item['metric_srid']}"
            )


def compute_lod1_orientation_metrics(
    conn, cities=None, target_table=SUMMARY_TABLE, verbose=True
):
    """按当前汇总表成员重算平均方向与方向熵。"""
    target_table = _require_metric_table(conn, target_table)
    plan = validate_lod1_metric_environment(conn, cities, require_summary=False)
    for item in plan:
        params = {
            "display_city": item["display_city"],
            "metric_srid": item["metric_srid"],
        }
        try:
            with conn.cursor() as cur:
                cur.execute(f"""
                    UPDATE {target_table}
                    SET avg_orientation=NULL, orientation_entropy=NULL
                    WHERE city=%(display_city)s;
                """, params)
                cur.execute(
                    build_lod1_orientation_update_sql(
                        item["city_code"], target_table=target_table
                    ),
                    params,
                )
                updated = cur.rowcount
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        if verbose:
            print(
                f"[{item['city_code']}] 朝向：{updated:,} 个街区，"
                f"EPSG:{item['metric_srid']}"
            )


def compute_avg_nn_distance_lod1(
    conn, cities=None, bld_table_schema="lod1",
    target_table=SUMMARY_TABLE, verbose=True
):
    """按城市测度 CRS 计算街区内建筑的平均最近邻距离（米）。"""
    if bld_table_schema != "lod1":
        raise ValueError("本函数仅支持 bld_table_schema='lod1'")
    target_table = _require_metric_table(conn, target_table)
    plan = validate_lod1_metric_environment(conn, cities, require_summary=False)
    total_computed = 0
    iterator = tqdm(plan, desc="NN Distance", unit="city", disable=not verbose)
    for item in iterator:
        city = item["city_code"]
        params = {
            "display_city": item["display_city"],
            "metric_srid": item["metric_srid"],
        }
        try:
            df = pd.read_sql_query(
                build_lod1_nn_centroid_query(city, target_table=target_table),
                conn,
                params=params,
            )
            update_data = []
            for block_id, group in df.groupby("block_id"):
                coords = group[["cx", "cy"]].to_numpy(dtype=float)
                value = _compute_block_nn_distance(coords)
                if value is not None and math.isfinite(value):
                    update_data.append((block_id, value))
            with conn.cursor() as cur:
                cur.execute(f"""
                    UPDATE {target_table} SET avg_nn_distance=NULL
                    WHERE city=%(display_city)s;
                """, params)
                if update_data:
                    execute_values(cur, f"""
                        UPDATE {target_table} AS v
                        SET avg_nn_distance=x.value
                        FROM (VALUES %s) AS x(block_id,value)
                        WHERE v.block_id=x.block_id::VARCHAR;
                    """, update_data, template="(%s, %s::numeric)", page_size=5000)
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        total_computed += len(update_data)
        if verbose:
            tqdm.write(
                f"[{city}] 最近邻：{len(update_data):,} 个街区，"
                f"EPSG:{item['metric_srid']}"
            )

    display_cities = [item["display_city"] for item in plan]
    with conn.cursor() as cur:
        cur.execute(f"""
            SELECT COUNT(*), COUNT(avg_nn_distance),
                   COUNT(*)-COUNT(avg_nn_distance),
                   AVG(avg_nn_distance), MIN(avg_nn_distance), MAX(avg_nn_distance)
            FROM {target_table} WHERE city=ANY(%s);
        """, (display_cities,))
        row = cur.fetchone()
    return {
        "total_blocks": int(row[0]), "computed": int(row[1]),
        "still_null": int(row[2]),
        "global_avg_m": float(row[3]) if row[3] is not None else None,
        "min_m": float(row[4]) if row[4] is not None else None,
        "max_m": float(row[5]) if row[5] is not None else None,
        "updated_this_run": total_computed,
    }


def compute_all_lod1_metrics(
    conn, cities=None, target_table=SUMMARY_TABLE,
    min_footprint_area_m2=1.0, verbose=True
):
    """在汇总表重建后，按依赖顺序重算全部汇总指标。"""
    compute_lod1_density_height_volume_metrics(
        conn, cities, target_table=target_table, verbose=verbose
    )
    compute_lod1_shape_metrics(
        conn, cities, target_table=target_table,
        min_footprint_area_m2=min_footprint_area_m2,
        verbose=verbose,
    )
    compute_lod1_orientation_metrics(
        conn, cities, target_table=target_table, verbose=verbose
    )
    nn_result = compute_avg_nn_distance_lod1(
        conn, cities, target_table=target_table, verbose=verbose
    )
    return {"nn_distance": nn_result}


# 兼容此前 notebook 的旧函数名。
compute_all_lod1_summary_metrics = compute_all_lod1_metrics


def summarize_lod1_metric_completeness(conn, target_table=SUMMARY_TABLE):
    """只读返回各城市指标非空计数。"""
    target_table = _require_metric_table(conn, target_table)
    fields = [
        "area_m2", "elongation", "compactness", "lod1_building_count", "bcr",
        "convexity", "rectangularity", "far", "bcd", "osr", "avh", "avh_w",
        "hsd", "hcv", "h_max", "h_range", "total_volume", "avg_volume",
        "avg_footprint", "footprint_cv", "avg_complexity", "avg_rectangularity",
        "avg_convexity", "avg_orientation", "orientation_entropy", "avg_nn_distance",
    ]
    counts = ",\n".join(f"COUNT({field}) AS {field}" for field in fields)
    return pd.read_sql_query(f"""
        SELECT city, COUNT(*) AS total, {counts}
        FROM {target_table} GROUP BY city ORDER BY city;
    """, conn)
