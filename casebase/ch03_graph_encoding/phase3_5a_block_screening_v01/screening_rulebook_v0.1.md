# 街区筛选规则书 v0.1

实验ID：`CH3-P3.5A-BLOCK-SCREEN-V01`
母样本：`summary.lod1_candidate_blocks`
分析单位：街区

## 状态优先级

`exclude_hard > review_scope > keep_provisional`。每行恰好取得一个机器状态；人工判断字段始终留空。

## 硬规则

- H01 `key_or_block_geometry_invalid`：标识为空、`city + block_id`重复，或街区几何为空/无效/面积非正。
- H02 `candidate_contract_broken`：`lod1_building_count < 1`，或违反既有`lod1_candidate_qa.py`中severity=`error`的`DOMAIN_RULES`或`CROSS_FIELD_RULES`。公式逐项等价冻结，不引入新阈值。
- H03 `coverage_inconsistent`：`bcr > 1`或`osr < 0`；同一街区只形成一个H03理由。
- H04 `unexpected_metric_missing`：单建筑街区除`hsd`、`hcv`、`footprint_cv`、`avg_nn_distance`外仍有指标缺失，或多建筑街区任一指标缺失。
- H05 `duplicate_centroid_signal`：多建筑街区且`avg_nn_distance == 0`。

H01至H05按顺序确定`primary_hard_reason`，同时保留全部命中原因。既有QA warning级覆盖率和零最近邻规则不再进入H02。

## 软复核规则

阈值只从未命中硬规则的街区计算：S01面积P99及以上、S02伸长度P01及以下、S03紧凑度P01及以下、S04建筑数P99及以上、S05 BCR P01及以下、S06同时命中S01与S05。边界取值使用包含式比较。

五项指标同时记录全局百分位和城市内百分位；百分位只用于解释，不产生额外删除规则。所有soft命中均为`review_scope`。

## 明确排除在v0.1之外

不使用Isolation Forest、LOF、聚类、固定污染率、建筑级buffer失败率、Tessellation失败率、弱边比例、城市特定阈值或按图规模直接删除。
