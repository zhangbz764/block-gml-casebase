# Phase 6A 模型输入合同 v0.1

状态：代码准备；未训练。源版本为v0.3.2；不修改Phase 5的103项注册表。逐字段实际路径、单位与目标冲突见`model_input_fields_v01.csv`。

## 公平比较的三组输入

| 组别 | 街区及条件 | 建筑 | 关系 |
|---|---|---|---|
| 街区指标＋共同条件 | Block15＋面积、编码建筑数、坐标尺度 | 无 | 无 |
| 建筑集合 | 同上 | Node8＋3项归一化派生值 | 无 |
| 关系图 | 同上 | 与集合完全相同 | 原邻接＋Edge12 |

Block实际路径为`block_attributes.values.xxx`，不是JSON顶层`block`。15项为elongation、compactness、bcr、convexity、rectangularity、far、bcd、osr、avh_w、hcv、footprint_cv、avg_complexity、avg_rectangularity、avg_convexity、orientation_entropy。`avh_w`是米制绝对高度，`bcd`是建筑数/公顷；不能将所有Block输入统称为尺度无关。

共同条件：`block_attributes.values.area_m2`、有效`len(nodes)`（对应`graph.node_count`）、`graph.coordinate_scale_m`。Node原8项含相对位置和朝向，集合也具有空间信息，不称其“完全不描述空间”。

Node新增的三个输入只作算术派生，不重算几何：轮廓面积/街区面积，高度/坐标尺度，近似体量/(街区面积×坐标尺度)。近似体量=轮廓面积×高度，与现有边体量比口径一致，不是精确三维体积。面积、高度、近似体量原值保留在`raw_values`供解释，默认不再重复拼入特征。集合和图共享派生值及缺失掩码。

## 缺失、顺序与身份

源null、非有限数值或明确的missing标记一律保留为`None`，掩码`True`表示缺失；派生分母缺失、非有限或≤0也保留null，不能用0高度替代缺失。原8/12项的源缺失原因不改写；派生缺失有说明。训练时如需插补、对数或缩放，只在训练集拟合，并在比较组间同样处理。本阶段不拟合缩放器或输出全量张量。

节点按`str(building_id)`排序，ID只作映射与追溯；边按`edge_id`排序，source/target重新映射到排序后节点下标。边索引当前保存为E×2列表，无边时为空；接入框架时可转2×E。Graph原无向边保留一次；`expand_bidirectional`可生成正反两向，必须交换`connection_axis_alignment_i/j`及对应掩码：i对应source，j对应target。其他现有12项为对称量。单节点与缺失Cell不减少有效Node。

城市、国家、ID、投影绝对坐标、数据源、审计状态、弱边flag、Cell failure均不是默认数值输入，仅作追溯/解释。掩码可用于处理缺失，但可能反映数据源差异，后续须单独检查这种混淆，不直接当作形态价值。

## 目标冲突与使用边界

字段表`target_conflict`是训练前提醒，不自动决定标签。预测FAR、avh_w或任一已输入量时，须在所有组剔除该目标及可直接重构该目标的输入，或改用真正涉及组织关系的目标。例如预测平均高度时，不仅删除Block.avh_w，还须考虑逐建筑高度及体量、面积条件的直接泄漏。不能预测输入自身并据此证明编码优越。

当前主任务仅条件性推荐生成规则家族辨认：接口、标签和数量/单栋形状等混淆尚未确认，`ready_for_phase6b=false`。真实候选页面不是训练标签集；开发池已查看，不可冒充未见评估集。本阶段只对入选Graph提取输入检查共享性，不保存模型文件，不训练，也不由关系摘要构造新特征。
