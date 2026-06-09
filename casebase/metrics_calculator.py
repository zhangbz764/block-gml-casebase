"""
Block 级指标计算工具库。

存放需要一定复杂计算的指标函数（相较于一两条公式或基础 SQL 就能解决的指标）。
函数按适用 LOD 层级命名或组织，方便扩展。

当前包含（Phase 7）：
    compute_avg_nn_distance(conn, verbose=True)
        Phase 7: 平均最近邻距离（m）
        用 Python + KDTree 替代 SQL CROSS JOIN LATERAL，避免 O(n²) 自连接。
        建筑质心投影到 Web Mercator (3857) 后在米制坐标空间计算。

使用方式（在 notebook 中）：
    from metrics_calculator import compute_avg_nn_distance
    result = compute_avg_nn_distance(conn)

所有函数统一约定：
    - 接收现有数据库连接 conn 作为第一个参数，不自行管理连接生命周期
    - 返回 dict 形式的验证结果统计
"""

from psycopg2.extras import execute_values
import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from tqdm import tqdm


# ═══════════════════════════════════════════════════════════════════
#  Phase 7: 平均最近邻距离 (Average Nearest Neighbor Distance)
#  适用 LOD: lod1, lod2
# ═══════════════════════════════════════════════════════════════════

def _compute_block_nn_distance(coords):
    """
    对单个 Block 内所有建筑的质心坐标，计算平均最近邻距离。

    Parameters
    ----------
    coords : ndarray of shape (n, 2)
        Web Mercator 投影后的米制坐标。

    Returns
    -------
    float or None
        平均最近邻距离（米），建筑数 < 2 时返回 None。
    """
    if len(coords) < 2:
        return None

    tree = KDTree(coords)
    # k=2: 第1近邻是自身(距离=0)，第2近邻是真正的最近邻
    distances, _ = tree.query(coords, k=2)
    nn_distances = distances[:, 1]  # 取每栋建筑的最近邻距离
    return float(np.mean(nn_distances))


def compute_avg_nn_distance_lod1(conn, cities=None, bld_table_schema="lod1", verbose=True):
    """
    计算所有城市的建筑平均最近邻距离（Phase 7）。

    对每个 Block 内 ≥2 栋建筑，用 KDTree 查询每栋建筑的最近邻距离，取 Block 均值。
    只有 1 栋建筑的 Block 保持 NULL。

    Parameters
    ----------
    conn : psycopg2 connection
        数据库连接（复用 notebook 已有的，不自行创建或关闭）。
    cities : list of str, optional
        城市名列表，每个值直接对应建筑表名的后缀（全小写）。
        例如 "newyork" → lod1.newyork_buildings_lod1。
        如果不传，则从 summary.lod1_valid_blocks 查询 DISTINCT city 作为默认值。
    bld_table_schema : str, default "lod1"
        建筑表所在的 schema，例如 "lod1" 或 "lod2"。
        表名规则为 {schema}.{city}_buildings_{schema}。
    verbose : bool, default True
        是否打印进度。

    Returns
    -------
    dict
        {
            "total_blocks": int,    # Block 总数
            "computed": int,        # 成功计算的数量
            "still_null": int,      # 仍为 NULL 的数量
            "global_avg_m": float,  # 全局均值（米）
            "min_m": float,         # 最小值（米）
            "max_m": float,         # 最大值（米）
        }
    """
    # ── Step 1: 获取城市列表 ──
    if cities is None:
        with conn.cursor() as cur:
            cur.execute("SELECT DISTINCT city FROM summary.lod1_valid_blocks ORDER BY city;")
            cities = [r[0] for r in cur.fetchall()]

    if verbose:
        print(f"共 {len(cities)} 个城市需要处理\n")

    total_computed = 0

    for city in tqdm(cities, desc="NN Distance", unit="city", disable=not verbose):
        bld_table = f"{bld_table_schema}.{city}_buildings_{bld_table_schema}"

        try:
            # 2a. 检查建筑表是否存在
            tbl_name = f"{city}_buildings_{bld_table_schema}"
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables
                        WHERE table_schema = %s AND table_name = %s
                    );
                """, (bld_table_schema, tbl_name))
                if not cur.fetchone()[0]:
                    tqdm.write(f"  [{city}] ✗ 表 {bld_table} 不存在，跳过")
                    continue

            # 2b. 拉取建筑质心坐标（投影到 3857 得到米制单位）
            df = pd.read_sql(f"""
                SELECT
                    building_id,
                    block_id,
                    ST_X(ST_Transform(ST_Centroid(geom_2d), 3857)) AS cx,
                    ST_Y(ST_Transform(ST_Centroid(geom_2d), 3857)) AS cy
                FROM {bld_table}
                WHERE block_id IS NOT NULL
                  AND geom_2d IS NOT NULL
                  AND ST_IsValid(geom_2d) = true;
            """, conn)

            if df.empty:
                tqdm.write(f"  [{city}] 无有效建筑几何，跳过")
                continue

            n_buildings = len(df)
            n_blocks_raw = df["block_id"].nunique()
            tqdm.write(f"  [{city}] 建筑: {n_buildings}, Block: {n_blocks_raw}")

            # 2c. 查询当前城市在汇总表中的有效 block_id
            with conn.cursor() as cur:
                cur.execute("SELECT block_id FROM summary.lod1_valid_blocks;")
                valid_block_ids = {r[0] for r in cur.fetchall()}

            # 2d. 按 block 分组，只保留在汇总表中的有效 block，用 KDTree 计算
            update_data = []
            for block_id, group in df.groupby("block_id"):
                if block_id not in valid_block_ids:
                    continue
                coords = group[["cx", "cy"]].values
                avg_nn = _compute_block_nn_distance(coords)
                if avg_nn is not None:
                    update_data.append((block_id, avg_nn))

            # 2e. 批量写回汇总表
            if update_data:
                with conn.cursor() as cur:
                    execute_values(cur, """
                        UPDATE summary.lod1_valid_blocks
                        SET avg_nn_distance = v.val
                        FROM (VALUES %s) AS v(block_id, val)
                        WHERE lod1_valid_blocks.block_id = v.block_id::VARCHAR;
                    """, update_data, template="(%s, %s::numeric)")
                conn.commit()
                total_computed += len(update_data)
                tqdm.write(f"  [{city}] ✓ 更新 {len(update_data)}/{n_blocks_raw} 个block")
            else:
                tqdm.write(f"  [{city}] 所有block均 <2 栋建筑，无需更新")

        except Exception as e:
            conn.rollback()
            tqdm.write(f"  [{city}] ✗ 错误: {e}")

    # ── Step 3: 验证结果 ──
    with conn.cursor() as cur:
        cur.execute("""
            SELECT
                COUNT(*)                                AS total_blocks,
                COUNT(avg_nn_distance)                  AS computed,
                COUNT(*) - COUNT(avg_nn_distance)        AS still_null,
                ROUND(AVG(avg_nn_distance)::numeric, 2)  AS global_avg_m,
                ROUND(MIN(avg_nn_distance)::numeric, 2)  AS min_m,
                ROUND(MAX(avg_nn_distance)::numeric, 2)  AS max_m
            FROM summary.lod1_valid_blocks;
        """)
        row = cur.fetchone()

    result = {
        "total_blocks": int(row[0]),
        "computed": int(row[1]),
        "still_null": int(row[2]),
        "global_avg_m": float(row[3]) if row[3] is not None else None,
        "min_m": float(row[4]) if row[4] is not None else None,
        "max_m": float(row[5]) if row[5] is not None else None,
    }

    if verbose:
        print("\n" + "=" * 60)
        print("  验证: avg_nn_distance 计算结果")
        print("=" * 60)
        print(f"  Block总数: {result['total_blocks']}")
        print(f"  已计算:    {result['computed']}")
        print(f"  仍为NULL:  {result['still_null']}")
        print(f"  全局均值:  {result['global_avg_m']} m")
        print(f"  最小值:    {result['min_m']} m")
        print(f"  最大值:    {result['max_m']} m")
        print(f"\nPhase 7 全部完成 (共更新 {total_computed} 个block)")

    return result