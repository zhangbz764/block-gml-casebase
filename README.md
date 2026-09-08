# Block-GML-Casebase — 街区级 3D 建筑案例数据处理

本仓库记录了处理多源 3D 城市建筑数据的过程，涵盖道路围合街区生成、CityJSON / CityGML 等格式的解析、PostgreSQL/PostGIS 入库、建筑与街区的空间叠合、LOD1 街区指标计算及街区级三维可视化。

## 处理流程

```
OpenStreetMap 道路 → 道路围合街区
多格式 LOD1/LOD2 建筑 → 标准化入库 → 建筑—街区匹配
道路街区 + LOD1 建筑 → 候选集 → 统一形态指标 → 质量检查 → 后续筛选与分析
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
| `lod1_metrics_pipeline.ipynb` | 当前 LOD1 候选街区汇总与统一指标计算入口 |
| `metrics_calculator.py` | 当前 LOD1 指标公式、候选表构建和批量计算的权威实现 |
| `block_unit_viz_lod1.py` | 街区单元 LoD1 建筑三维可视化 |
| `block_unit_viz_lod2.py` | 街区单元 LoD2 建筑三维可视化 |
| `block_unit_viz_switch.py` | LoD1/LoD2 切换可视化 |
| `sql_utils.py` | 数据库表结构创建与建筑-街区空间叠合 |
| `gdf_parser.py` | GeoDataFrame 解析辅助 |
| `utils_z.py` | 命令行执行与数据库连接等基础工具 |

## 当前 LOD1 指标入口

当前基础指标流程只运行：

```text
casebase/lod1_metrics_pipeline.ipynb
    └─ casebase/metrics_calculator.py
```

旧 `block_metric_check.ipynb`、`block_summary_metrics.ipynb` 以及指标重建前的聚类 notebook 已移动到 `casebase/archive/`。这些文件用于历史追溯，不属于现行运行顺序。

## 输出数据

解析结果存入 PostgreSQL。数据获取和匹配阶段的记录规模如下；不同列对应不同处理口径，不能直接作为最终分析样本量：

| 级别 | 数据城市数 | 匹配到街区的建筑数 | 历史记录中的含建筑街区数 |
|---|---:|---:|---:|
| LOD1 | 25 | 12,187,282 | 261,978 |
| LOD2 | 19 | 4,331,859 | 79,550 |

当前 `summary.lod1_candidate_blocks` 从24个具有可用 LOD1 数据的城市汇总得到219,419个街区，候选条件为至少匹配一栋 LOD1 建筑。该表尚未实施最终异常筛选。

## 目录说明

- `casebase/archive/`：已执行修复、被替代流程和旧版分析实验；参见其中的 `README.md`；
- `casebase/quality_checks/`：候选指标质检程序、修复记录和质检输出；
- `casebase/20260903_record_text/`：代码理解记录、技术问题总表和论文工作稿；
- `casebase/statistics/`：逐城市数据统计与记录；
- `casebase/output/`：历史图件和分析输出。
