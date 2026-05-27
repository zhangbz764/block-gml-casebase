"""
AUTHOR: zhangbz
PROJECT: UrbanPlayground
DATE: 2026/4/26
TIME: 17:52
DESCRIPTION: This module provides functions to parse CityJSON files, specifically for LOD1 building data.
"""

import json
import attrs
import numpy as np
from shapely.geometry import Polygon
from shapely.wkt import dumps as wkt_dumps
from lxml import etree


# ── 工具函数：统一高度和层数验证逻辑 ──

def _validate_height(height, ground_z, roof_face):
    """
    确保 height > 0，否则用 roof_face 的 Z 坐标重算。
    返回有效的 height 值，或 None（无法确定有效高度时丢弃该建筑）。
    """
    if height is not None and height > 0:
        return float(height)

    # height 为 None / <= 0，尝试用 Z 坐标
    if roof_face is not None:
        top_z = float(np.mean([c[2] for c in roof_face.exterior.coords]))
        h = top_z - ground_z
        if h > 0:
            return h
    return None


def _validate_floor_count(floor_count, height):
    """
    确保 floor_count 是正整数，否则用 height/3 估算。
    返回有效的 floor_count 值，或 0（无法确定时）。
    """
    if floor_count is not None:
        try:
            fc = int(floor_count)
            if fc > 0:
                return fc
        except (ValueError, TypeError):
            pass

    # 无效，用 height/3 估算
    if height is not None and height > 0:
        return max(round(height / 3), 1)
    return 0


# ── 通用解析器 ──

def parse_cityjson_lod1(filepath, target_lod="1"):
    """解析CityJSON文件并提取LOD1建筑物数据。"""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 空文件检查
    if not data.get("CityObjects") or not data.get("vertices"):
        print(f"空文件，跳过：{filepath}")
        return []

    scale = np.array(data["transform"]["scale"])
    if np.all(scale == 0):
        print(f"无效scale，跳过：{filepath}")
        return []
    translate = np.array(data["transform"]["translate"])
    vertices = np.array(data["vertices"])
    real_vertices = vertices * scale + translate

    buildings = []
    no_lod1_count = 0
    for obj_id, obj in data["CityObjects"].items():
        if obj["type"] not in ("Building", "BuildingPart"):
            continue

        attrs = obj.get("attributes", {})

        height_raw = attrs.get("measuredHeight")
        floor_count_raw = attrs.get("storeysAboveGround")
        function = attrs.get("function")

        geom_entry = next((g for g in obj.get("geometry", []) if str(g.get("lod")) == target_lod), None)
        if geom_entry is None:
            no_lod1_count += 1
            continue

        faces = []
        for shell in geom_entry["boundaries"]:
            for face in shell:
                ring = face[0]
                coords = [tuple(real_vertices[i]) for i in ring]
                if len(coords) >= 3:
                    faces.append(Polygon(coords))

        if not faces:
            print(f"No valid faces for building: {obj_id}")
            continue

        surfaces = classify_surfaces(faces)

        # 直接从分类结果里取底面和顶面
        ground_face = next((f for stype, f in surfaces if stype == "GroundSurface"), None)
        roof_face = next((f for stype, f in surfaces if stype == "RoofSurface"), None)

        if ground_face is None:
            print(f"No ground face for building: {obj_id} >>> {filepath}")
            continue

        ground_z = float(np.mean([c[2] for c in ground_face.exterior.coords]))

        # 统一验证高度（≤0 则用 Z 坐标重算）
        height = _validate_height(height_raw, ground_z, roof_face)
        if height is None:
            print(f"Invalid height for building: {obj_id}")
            continue

        # 统一处理层数
        floor_count = _validate_floor_count(floor_count_raw, height)

        geom_2d = Polygon([(c[0], c[1]) for c in ground_face.exterior.coords])

        buildings.append({
            "citygml_id": obj_id,
            "height": height,
            "ground_z": ground_z,
            "floor_count": floor_count,
            "function": function,
            "geom_2d": geom_2d,
            "surfaces": surfaces
        })
    return buildings


# ── INSERT 函数 ──

def insert_buildings_lod1(buildings, conn, lod1_table, surface_table,
                           city_prefix, target_srid, source_srid,
                           building_counter, surface_counter):
    """批量插入LOD1建筑物和表面数据到数据库，支持坐标系转换。"""
    cur = conn.cursor()

    if source_srid == target_srid:
        geom_expr = f"ST_GeomFromText(%s, {target_srid})"
    else:
        geom_expr = f"ST_Transform(ST_GeomFromText(%s, {source_srid}), {target_srid})"

    sql_building = f"""
        INSERT INTO {lod1_table}
            (building_id, citygml_id, geom_2d, height, ground_z, floor_count, function)
        VALUES (%s, %s, {geom_expr}, %s, %s, %s, %s)
        ON CONFLICT (building_id) DO NOTHING;
    """

    sql_surface = f"""
        INSERT INTO {surface_table}
            (surface_id, building_id, surface_type, geom_3d)
        VALUES (%s, %s, %s, {geom_expr})
    """

    building_rows = []
    surface_rows = []

    for b in buildings:
        building_id = f"{city_prefix}_B_{str(building_counter).zfill(7)}"
        building_counter += 1

        building_rows.append((
            building_id,
            b["citygml_id"],
            wkt_dumps(b["geom_2d"]),
            b["height"],
            b["ground_z"],
            b["floor_count"],
            b["function"]
        ))

        for stype, poly in b["surfaces"]:
            surface_id = f"{city_prefix}_S_{str(surface_counter).zfill(8)}"
            surface_counter += 1
            surface_rows.append((
                surface_id,
                building_id,
                stype,
                wkt_dumps(poly)
            ))

    cur.executemany(sql_building, building_rows)
    cur.executemany(sql_surface, surface_rows)
    conn.commit()
    cur.close()

    return len(building_rows), building_counter, surface_counter


# ── 阿姆斯特丹（荷兰）──

def parse_cityjson_lod1_NL_AM(filepath, target_lod="1.3"):
    """解析阿姆斯特丹CityJSON数据，从BuildingPart继承父级Building属性。"""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    scale         = np.array(data["transform"]["scale"])
    translate     = np.array(data["transform"]["translate"])
    vertices      = np.array(data["vertices"])
    real_vertices = vertices * scale + translate

    # 先建立Building属性查找表
    building_attrs = {}
    for obj_id, obj in data["CityObjects"].items():
        if obj["type"] == "Building":
            building_attrs[obj_id] = obj.get("attributes", {})

    buildings = []
    for obj_id, obj in data["CityObjects"].items():
        if obj["type"] != "BuildingPart":
            continue

        # 从父级Building取属性
        parent_id = obj_id.rsplit("-", 1)[0]
        attrs     = building_attrs.get(parent_id, {})

        height_raw = attrs.get("b3_h_dak_50p")
        h_maaiveld = attrs.get("b3_h_maaiveld")
        floor_count_raw = attrs.get("b3_bouwlagen")
        function    = attrs.get("status")

        geom_entry = next(
            (g for g in obj.get("geometry", []) if str(g.get("lod")) == target_lod),
            None
        )
        if geom_entry is None:
            continue

        # LOD1: 直接从boundaries重建faces
        faces = []
        for shell in geom_entry["boundaries"]:
            for face in shell:
                ring   = face[0]
                coords = [tuple(real_vertices[i]) for i in ring]
                if len(coords) >= 3:
                    faces.append(Polygon(coords))

        if not faces:
            print(f"No valid faces for building: {obj_id}")
            continue

        surfaces = classify_surfaces(faces)

        ground_face = next((f for stype, f in surfaces if stype == "GroundSurface"), None)
        roof_face   = next((f for stype, f in surfaces if stype == "RoofSurface"), None)

        if ground_face is None:
            print(f"No ground face for building: {obj_id}")
            continue

        ground_z = float(np.mean([c[2] for c in ground_face.exterior.coords]))

        # 阿姆斯特丹特殊处理：净高度 = 屋顶高程 - 地面高程
        if height_raw is not None and h_maaiveld is not None:
            height = float(height_raw) - float(h_maaiveld)
        else:
            height = None

        # 统一验证高度（≤0 则用 Z 坐标重算）
        height = _validate_height(height, ground_z, roof_face)
        if height is None:
            print(f"Invalid height for building: {obj_id}")
            continue

        # 统一处理层数
        floor_count = _validate_floor_count(floor_count_raw, height)

        geom_2d = Polygon([(c[0], c[1]) for c in ground_face.exterior.coords])

        buildings.append({
            "citygml_id":  obj_id,
            "height":      height,
            "ground_z":    ground_z,
            "floor_count": floor_count,
            "function":    function,
            "geom_2d":     geom_2d,
            "surfaces":    surfaces
        })

    return buildings


# ── 美国 ──

def parse_cityjson_lod1_US(filepath, target_lod="1"):
    """解析美国CityJSON数据，处理可选的transform字段和相对高度。"""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 空文件检查（最先）
    if not data.get("CityObjects") or not data.get("vertices"):
        print(f"空文件，跳过：{filepath}")
        return []

    # transform判断
    if "transform" in data:
        scale = np.array(data["transform"]["scale"])
        if np.all(scale == 0):
            print(f"无效scale，跳过：{filepath}")
            return []
        translate = np.array(data["transform"]["translate"])
        real_vertices = np.array(data["vertices"]) * scale + translate
    else:
        real_vertices = np.array(data["vertices"])

    buildings = []
    no_lod1_count = 0
    for obj_id, obj in data["CityObjects"].items():
        if obj["type"] not in ("Building", "BuildingPart"):
            continue

        attrs = obj.get("attributes", {})

        height_raw = attrs.get("measuredHeight")
        floor_count_raw = attrs.get("storeysAboveGround")
        function = attrs.get("function")

        geom_entry = next((g for g in obj.get("geometry", []) if str(g.get("lod")) == target_lod), None)
        if geom_entry is None:
            no_lod1_count += 1
            continue

        faces = []
        for shell in geom_entry["boundaries"]:
            for face in shell:
                ring = face[0]
                coords = [tuple(real_vertices[i]) for i in ring]
                if len(coords) >= 3:
                    faces.append(Polygon(coords))

        if not faces:
            print(f"No valid faces for building: {obj_id}")
            continue

        surfaces = classify_surfaces_flat(faces)

        # 直接从分类结果里取底面和顶面
        ground_face = next((f for stype, f in surfaces if stype == "GroundSurface"), None)
        roof_face = next((f for stype, f in surfaces if stype == "RoofSurface"), None)

        if ground_face is None:
            print(f"No ground face for building: {obj_id} >>> {filepath}")
            continue

        ground_z = float(np.mean([c[2] for c in ground_face.exterior.coords]))

        # 统一验证高度（≤0 则用 Z 坐标重算）
        height = _validate_height(height_raw, ground_z, roof_face)
        if height is None:
            print(f"Invalid height for building: {obj_id}")
            continue

        # 统一处理层数
        floor_count = _validate_floor_count(floor_count_raw, height)

        geom_2d = Polygon([(c[0], c[1]) for c in ground_face.exterior.coords])

        buildings.append({
            "citygml_id": obj_id,
            "height": height,
            "ground_z": ground_z,
            "floor_count": floor_count,
            "function": function,
            "geom_2d": geom_2d,
            "surfaces": surfaces
        })
    return buildings


# ── 日本 ──

# 命名空间
NS = {
    "bldg": "http://www.opengis.net/citygml/building/2.0",
    "gml":  "http://www.opengis.net/gml",
    "core": "http://www.opengis.net/citygml/2.0",
}

def parse_citygml_lod1_JP(filepath):
    tree = etree.parse(filepath)
    root = tree.getroot()

    buildings = []

    for bldg_el in root.iter("{http://www.opengis.net/citygml/building/2.0}Building"):
        obj_id = bldg_el.get("{http://www.opengis.net/gml}id")

        # 属性
        def get_attr(tag):
            el = bldg_el.find(f".//bldg:{tag}", NS)
            return el.text if el is not None else None

        height_raw   = get_attr("measuredHeight")
        floor_count_raw = get_attr("storeysAboveGround")
        function    = get_attr("usage")

        height_raw   = float(height_raw) if height_raw is not None else None
        # 原逻辑 floor_count != "9999" 现在由 _validate_floor_count 统一处理

        # LOD1几何
        solid_el = bldg_el.find(".//bldg:lod1Solid/gml:Solid", NS)
        if solid_el is None:
            continue

        faces = []
        for poslist_el in solid_el.iter("{http://www.opengis.net/gml}posList"):
            vals = list(map(float, poslist_el.text.strip().split()))
            # 日本格式：纬度 经度 Z，每3个一组
            coords = []
            for i in range(0, len(vals) - 2, 3):
                lat, lon, z = vals[i], vals[i+1], vals[i+2]
                coords.append((lon, lat, z))  # 交换为经度/纬度
            if len(coords) >= 3:
                faces.append(Polygon(coords))

        if not faces:
            continue

        surfaces = classify_surfaces_flat(faces)

        ground_face = next((f for stype, f in surfaces if stype == "GroundSurface"), None)
        roof_face   = next((f for stype, f in surfaces if stype == "RoofSurface"), None)

        if ground_face is None:
            print(f"No ground face: {obj_id}")
            continue

        ground_z = float(np.mean([c[2] for c in ground_face.exterior.coords]))

        # 统一验证高度（≤0 则用 Z 坐标重算）
        height = _validate_height(height_raw, ground_z, roof_face)
        if height is None:
            print(f"Invalid height for building: {obj_id}")
            continue

        # 统一处理层数（自动过滤无效值如 "9999"）
        floor_count = _validate_floor_count(floor_count_raw, height)

        geom_2d = Polygon([(c[0], c[1]) for c in ground_face.exterior.coords])

        buildings.append({
            "citygml_id":  obj_id,
            "height":      height,
            "ground_z":    ground_z,
            "floor_count": floor_count,
            "function":    function,
            "geom_2d":     geom_2d,
            "surfaces":    surfaces
        })

    return buildings


# ── 辅助函数：法向量计算和面分类 ──

def get_normal(poly):
    """计算多边形的法向量，用于识别表面朝向。"""
    pts = np.array(poly.exterior.coords[:-1])
    if len(pts) < 3 or pts.shape[1] != 3:
        return None
    v0 = pts[0]
    v1 = None
    for p in pts[1:]:
        if np.linalg.norm(p - v0) > 1e-6:
            v1 = p
            break
    if v1 is None:
        return None
    for p in pts[1:]:
        candidate = np.cross(v1 - v0, p - v0)
        if np.linalg.norm(candidate) > 1e-6:
            return candidate / np.linalg.norm(candidate)
    return None


def classify_surfaces(faces):
    """根据法向量将面分类为地面、屋顶和墙体。"""
    horizontal = []
    surfaces = []
    
    for face in faces:
        normal = get_normal(face)
        if normal is None:
            continue
        if abs(normal[2]) > 0.9:
            horizontal.append(face)
        else:
            surfaces.append(("WallSurface", face))
    
    if len(horizontal) >= 2:
        horizontal.sort(key=lambda p: np.mean([c[2] for c in p.exterior.coords]))
        surfaces.append(("GroundSurface", horizontal[0]))
        surfaces.append(("RoofSurface", horizontal[-1]))
    elif len(horizontal) == 1:
        z_mean = np.mean([c[2] for c in horizontal[0].exterior.coords])
        z_all = [np.mean([c[2] for c in f.exterior.coords]) for f in faces]
        stype = "GroundSurface" if z_mean < np.mean(z_all) else "RoofSurface"
        surfaces.append((stype, horizontal[0]))
    
    return surfaces


def classify_surfaces_flat(faces):
    """根据Z值分类面，适用于相对高度数据（Z为相对值，XY为经纬度）。"""
    z_means = [(np.mean([c[2] for c in f.exterior.coords]), f) for f in faces]
    z_values = [z for z, _ in z_means]
    z_min = min(z_values)
    z_max = max(z_values)
    
    surfaces = []
    for z_mean, face in z_means:
        if abs(z_mean - z_min) < 0.01:
            surfaces.append(("GroundSurface", face))
        elif abs(z_mean - z_max) < 0.01:
            surfaces.append(("RoofSurface", face))
        else:
            surfaces.append(("WallSurface", face))
    return surfaces