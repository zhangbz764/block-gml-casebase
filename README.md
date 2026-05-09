# Block-GML-Casebase — 街区级 3D 建筑案例数据处理

本仓库记录了处理多源 3D 城市建筑数据的过程，涵盖 CityJSON / CityGML 等格式的解析、PostgreSQL 入库、建筑与街区的空间叠合，以及街区级三维可视化。

## 处理流程

```
OpenStreetMap+CityJSON/CityGML → 解析 → PostgreSQL/PostGIS → 空间叠合（建筑→街区） → 3D 可视化
```

## 文件说明

| 文件 | 内容 |
|---|---|
| `city_block_extractor.ipynb` | 基于OpenStreetMap道路数据的街区（block）几何提取 |
| `gml_process_lod1.ipynb` | CityGML/CityJSON LoD1 建筑数据的解析与入库 |
| `gml_process_lod2.ipynb` | CityGML/CityJSON LoD2 建筑数据的解析与入库 |
| `cityjson_parser_lod1.py` | CityJSON LoD1 建筑数据的核心parser和insert |
| `cityjson_parser_lod2.py` | CityJSON LoD2 建筑数据的核心parser和insert |
| `shp_process.ipynb` | Shapefile 建筑数据的解析与入库 |
| `other_format_process.ipynb` | 其他城市模型格式（geojson、gpkg、parquet等）的处理尝试 |
| `block_unit_viz_lod1.py` | 街区单元 LoD1 建筑三维可视化 |
| `block_unit_viz_lod2.py` | 街区单元 LoD2 建筑三维可视化 |
| `block_unit_viz_switch.py` | LoD1/LoD2 切换可视化 |
| `sql_utils.py` | 数据库表结构创建与建筑-街区空间叠合 |
| `gdf_parser.py` | GeoDataFrame 解析辅助 |
| `utils_z.py` | 命令行执行与数据库连接等基础工具 |

## 输出数据

解析结果存入 PostgreSQL，截至 2026/05/09 的数据规模如下：

| 级别 | 城市数量 | 建筑数量 | 含建筑街区数量 |
|---|---|---|---|
| LoD1 | 25 | 12,191,248 | 261,978 |
| LoD2 | 19 | 4,331,859 | 79,550 |
