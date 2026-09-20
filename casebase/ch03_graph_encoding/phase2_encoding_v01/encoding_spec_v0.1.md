# 街区分层属性图编码规范 v0.1

状态：阶段2可执行原型  
编码版本：`0.1`  
对象：`summary.lod1_candidate_blocks`中的少量LOD1候选街区  
格式：每个街区一个UTF-8 JSON文件

## 1. 编码对象

每个街区编码为：

\[
\mathcal{E}_b=(\mathbf{u}_b,\mathcal{G}_b),\qquad
\mathcal{G}_b=(V_b,E_b,X_b,R_b)
\]

- `graph`保存街区身份、坐标系、局部变换、质量标记和源表索引；
- `block_attributes`保存候选表26项原始指标及逐项缺失标记；
- `nodes`保存建筑身份、几何派生属性、位置、缺失和质量状态；
- `edges`保存一条无向建筑关系及关系属性；
- `diagnostics`保存邻接构造状态、重复质心、连通分量和平均度。

JSON不依赖PyTorch Geometric。`building_id`、`citygml_id`、源表名和源几何列均保留，能够回查数据库。样本文件额外保存米制WKT，用于脱离数据库重绘；WKT属于核对与追溯字段，不是默认模型输入。

## 2. 街区级指标

完整保留阶段1核实的26项指标原值，不在本阶段筛选或标准化。JSON以`null`表示数据库NULL，并在`block_attributes.missing`中保存同名布尔标记。真实0值保留为0，不与缺失混淆。

26项指标是：`area_m2`、`elongation`、`compactness`、`lod1_building_count`、`bcr`、`convexity`、`rectangularity`、`far`、`bcd`、`osr`、`avh`、`avh_w`、`hsd`、`hcv`、`h_max`、`h_range`、`total_volume`、`avg_volume`、`avg_footprint`、`footprint_cv`、`avg_complexity`、`avg_rectangularity`、`avg_convexity`、`avg_orientation`、`orientation_entropy`和`avg_nn_distance`。

## 3. 坐标与尺度

1. 街区和建筑几何从EPSG:4326读取后，使用`summary.city_name_map.lod1_metric_srid`投影；
2. 局部原点取投影后街区几何的质心；
3. 尺度取`summary.lod1_candidate_blocks.area_m2`的平方根；
4. `local_x_m`与`local_y_m`为建筑底面质心相对局部原点的米制位移；
5. 归一化位置和距离除以同一尺度；
6. v0.1不旋转街区，保留北向；
7. 建筑长轴采用180度周期，保存`sin(2θ)`与`cos(2θ)`。

绝对米制值与无量纲归一化值同时保存。尺度必须有限且大于0，否则该街区不构图。

## 4. 建筑节点

节点以数据库`building_id`为稳定ID，同时保留`citygml_id`。代表点取投影后底面几何质心。

底面积和周长有两组：

- `source_area_m2`、`source_perimeter_m`是建筑表原值；
- `footprint_area_m2`、`footprint_perimeter_m`由投影后WKT重新计算，是v0.1几何派生的核心值。

派生形态字段：

- 紧凑度：`4πA/P²`；
- 矩形度：`A/A_minimum_rotated_rectangle`；
- 凸度：`A/A_convex_hull`；
- 长轴方向：最小旋转外接矩形最长边方向，范围`[0,180)`；
- 到街区边界距离：建筑质心到街区边界的最短欧氏距离。

正方形或近正方形底面的长轴不稳定时，方向值仍保存以保证可重绘，同时令`orientation_stable=false`。`function`、`ground_z`、层数、屋顶类型和建造年份不进入v0.1核心节点属性。

节点质量标记至少包括：几何有效、质心是否被街区覆盖、完整底面是否被街区覆盖、方向是否稳定。完整底面跨界不会被裁剪或删除。

## 5. 建筑关系边

v0.1以建筑代表点的Delaunay三角剖分生成一组无向边：

- 1节点：合法无边图，状态`single_node`；
- 2个不同代表点：一条无向边，状态`two_node_rule`；
- 3节点及以上：Delaunay边去重后保存，状态`delaunay`；
- 重复代表点：状态`degenerate_duplicate_points`，不抖动、不合并、不删点，不生成边；
- 共线点：状态`degenerate_collinear`，不生成边；
- Qhull失败：状态`degenerate_delaunay_failure`，不生成边。

重复点判断先于双节点规则，因此两个重合节点被视为显式退化，而不是生成零长度边。本阶段没有自动切换到k近邻。

边在JSON中始终按`building_id`字典序保存`source`和`target`，从而使有符号位移可重复。距离、底面最短距离、方位、方向差、高度差/比、底面积比同时保存。缺少任一前置属性时相应值为`null`，并在边的`missing`中标记；不以0代替。

## 6. 自动一致性规则

每个样本至少检查：

- JSON节点数等于候选表和建筑表记录数；
- `building_id`唯一且非空；
- 声明节点数等于实际节点数；
- 不存在自环或重复无向边；
- 距离非负；
- JSON中没有无说明的NaN或Infinity；
- 核心缺失标记与`null`一致；
- 坐标尺度有限且大于0；
- JSON重新读取后的节点ID和边集合完全一致；
- 邻接状态、连通分量、平均度、最长边和中位边长被记录。

边界样本可以以明确退化状态结束，不要求伪造可用边。

## 7. 建筑学判断与当前适用边界

核对图把原始底面、Delaunay边、建筑高度和归一化图并置。v0.1在密集、相对连续的布局中通常能保留局部邻近关系；在稀疏、强凹形、有大空地或离散组团的街区中，Delaunay会跨越空地或凹口产生长边。

该问题已在Winnipeg低覆盖样本和Portland凹形分叉样本中重复出现，并在若干Berlin、Tokyo和超大街区样本中有较弱表现。因此Delaunay在v0.1中是“可执行的基线关系”，不是已经确认的最终空间语义。进入阶段3前应决定是增加受边界约束/距离阈值规则，还是把长边质量标记交给后续关系增量验证。本阶段不进行多邻接算法比较。

## 8. 相对任务书的实现调整

- 增加`source_area_m2`和`source_perimeter_m`，并把投影几何重算值与源值区分；
- 增加`footprint_covered_by_block`，用于识别质心在街区内但完整底面跨界的对象；
- 样本JSON保存米制WKT，以支持无数据库重绘；
- 重复质心的双节点样本以退化状态优先，不建立零长度边；
- 没有引入k近邻备用图，因为退化只出现在指定边界样本，而更主要的问题是Delaunay长边语义，需要研究者先决定规则。

