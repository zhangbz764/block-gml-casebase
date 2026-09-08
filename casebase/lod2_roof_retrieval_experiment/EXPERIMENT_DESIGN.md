# Luxembourg LOD2 屋顶信息检索实验设计

## 1. 目标与当前范围

本实验检验：在街区总体尺度、开发强度、建筑体量和布局相近的条件下，加入 LOD2 屋顶信息，是否能检索到屋顶形态更接近的参照街区，以及这种变化需要付出多少总体形态可比性代价。

投稿前只处理 Luxembourg。当前阶段完成：最低数据质检、同源 G1/G2 表征、M1/M2 指标、开发查询、候选池和 Top-5 检索。正式评价、批量图面和评分方案后置。

## 2. 数据与同源对照

- 唯一源数据为 `lod2.luxembourg_buildings_lod2`、`lod2.luxembourg_building_surfaces_lod2` 及对应道路街区表。
- 实验数据库对象统一写入独立 schema `exp_lod2_roof_lu`；既有 `block`、`lod1`、`lod2` 和 `summary` schema 只读，不新增字段、不建临时表、不覆盖数据。
- G2 保留数据库中的 LOD2 屋顶几何。
- G1 使用同一批建筑对象和底面，将每栋建筑简化为单一高度体块。
- 既有 LOD1 指标的定义和实现思路可以复用，但所有数值必须从 Luxembourg LOD2 对象重新计算；不得把独立 LOD1 表中的历史数值混入同源实验。

## 3. G1 单一高度

G1 高度采用屋顶水平投影面积加权平均高程减去地面高程：

```text
h_g1 = sum(projected_area_i * mean_roof_z_i) / sum(projected_area_i) - ground_z
```

其中：

- `projected_area_i` 是屋顶面的水平投影面积；它用于近似体积等效高度，避免陡坡因三维面积更大而被额外放大。
- `mean_roof_z_i` 是该屋顶面的顶点平均高程。
- `ground_z` 优先采用所有 GroundSurface 顶点高程的中位数。
- 屋顶坡度的加权使用真实三维面积，与 G1 高度的水平投影面积权重严格分开。

## 4. 最低数据质量门槛

本实验不追求修复所有 CityGML/CityJSON 瑕疵。孔洞丢失、多个 RoofSurface/GroundSurface、建模粒度差异和非关键表面异常只记录为警告，不因其本身排除街区。

建筑级硬条件：

1. 已匹配到街区；
2. 有可读取的 GroundSurface 和 RoofSurface；
3. 能得到有限且为正的 `h_g1`；
4. 至少有一个可计算坡度和三维面积的有效屋顶三角形。

街区级硬条件：

1. 至少有一栋满足建筑级硬条件的建筑；
2. 可用建筑底面积占该街区全部已匹配建筑底面积的比例不低于 80%；
3. M1 和 M2 的必需指标均为有限值。

所有排除和警告保存原因码，不回写或删除原始数据库对象。

## 5. M1：LOD2 派生的总体形态基线

优先复用现有 LOD1 指标定义，并从 LOD2 建筑重算以下 12 项候选指标：

1. `log_block_area`：街区面积对数；
2. `compactness`：街区紧凑度；
3. `elongation`：最小旋转外接矩形短边/长边；
4. `bcr`：建筑底面积总和/街区面积；
5. `volume_density`：`sum(footprint_area * h_g1) / block_area`；
6. `mean_h_g1`：建筑 G1 高度均值；
7. `h_g1_cv`：建筑 G1 高度变异系数；
8. `log_building_count`：可用建筑对象数对数；
9. `mean_footprint_area`：平均建筑底面积；
10. `footprint_area_cv`：建筑底面积变异系数；
11. `orientation_entropy`：建筑轴向熵；
12. `normalized_nn_distance`：平均最近邻距离除以街区面积平方根。

不采用 FAR：当前层数包含按 3 m 层高估算的值，而且 FAR 与 BCR、建筑高度和体量密度高度相关。开发阶段检查缺失率、分布和 Spearman 相关性；若任意一对候选指标的绝对相关系数超过 0.90，优先保留定义更直接、缺失更少的一项，并在配置文件中记录最终列表。

## 6. M2：屋顶增量指标

M2 完整保留 M1，再加入三项街区级屋顶指标：

1. `roof_mean_slope`：全部有效屋顶三角形按真实三维面积加权的平均坡度；
2. `roof_relief_ratio_median`：逐建筑计算 `(roof_z_max - roof_z_min) / (roof_z_max - ground_z)`，再取街区中位数；
3. `building_roof_slope_sd`：逐建筑先求面积加权平均坡度，再计算街区内建筑间的总体标准差；单栋建筑街区记为 0。

屋顶多边形先转换到 Luxembourg 的 `lod2_metric_srid`，再三角化并计算面积与法向量。不得直接在 EPSG:4326 中计算坡度或面积，不使用屋顶面片数量作为复杂度指标。

## 7. 检索规则

1. 在全部可用街区上用固定的稳健缩放计算 M1 总体形态距离；
2. 对每个查询先取 M1 距离最近的 50 个街区，形成两种方法共享的总体形态候选池；
3. M1 返回候选池中总体形态距离最小的 5 个；
4. M2 在相同候选池中按组合距离重新排序：

```text
D_M2 = (1 - lambda) * D_general + lambda * D_roof
```

5. 开发查询比较 `lambda = 0.20、0.35、0.50`，依据候选的总体形态损失和屋顶区分度确定一个主值；正式查询不再调参。
6. 排除查询自身；同一 `block_id` 只保留一个候选记录。

## 8. 开发阶段输出

- `quality_flags.csv`：建筑/街区可用性、警告和排除原因；
- `building_metrics.parquet`：建筑级 G1 与屋顶指标；
- `block_metrics.parquet`：M1/M2 街区指标；
- `experiment_config.json`：城市、SRID、阈值、指标列表、缩放和权重；
- `development_queries.csv`：4 类屋顶情景各 1 个查询，另加 1 个边界案例；
- `development_retrieval_results.csv`：M1/M2 Top-5、距离分量和排名。

中间文件只写入本实验文件夹，不改动已有 LOD1 指标表和原始 LOD2 表。

数据库中对应保存 `exp_lod2_roof_lu.building_metrics`、`exp_lod2_roof_lu.block_metrics` 和 `exp_lod2_roof_lu.retrieval_results`。Notebook 可以重复创建或替换这三个实验表，但不得对其他 schema 执行 DDL 或 DML。

## 9. 代码组织

```text
lod2_roof_retrieval_experiment/
├─ EXPERIMENT_DESIGN.md
├─ 01_luxembourg_metrics_and_retrieval.ipynb
├─ roof_geometry.py
├─ block_metrics.py
├─ retrieval.py
└─ outputs/
```

- notebook 是可阅读、可分步执行的主流程；
- `roof_geometry.py` 负责投影、三角化、坡度、三维面积和 G1 高度；
- `block_metrics.py` 负责建筑级和街区级指标汇总；
- `retrieval.py` 负责缩放、候选池、距离与 Top-5；
- notebook 中不复制方法实现，只负责参数、调用、质检摘要和结果展示。

## 10. 文献策略

编码不以完成系统综述为前提。几何定义先采用可解释、可测试的标准方法；在摘要写作前再针对三类内容补充少量方法来源：LOD/CityGML 语义、三维屋顶/城市形态指标、基于特征距离的案例检索。若文献建议与已实现指标冲突，只在正式查询前调整并记录版本，不在看到正式结果后改指标。

## 11. 验证与失败处理

- 用人工构造的平屋顶、单坡屋顶和 45° 双坡屋顶验证坡度、三维面积、G1 高度和屋顶高差比；数值误差超过 `1e-6` 视为失败。
- 从数据库固定抽取至少 10 栋建筑，将程序结果与独立的顶点/Z 值手算或 SQL 汇总交叉检查。
- 单个表面无法三角化时记录原因并跳过该表面；单栋建筑或单个街区不满足最低覆盖率时记录原因并排除，不中断全流程。
- 每次运行固定随机种子，并保存查询 ID、候选池 ID、最终指标列表、缩放参数和权重。
- notebook 的每个主要阶段先显示记录数、缺失率和范围摘要，通过后才写出对应中间文件。
