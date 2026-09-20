# 阶段3确定性抽样规范 v0.3

实验ID：`CH3-P3-ENCODING-SCALE-V03`。抽样单位为只读关系`summary.lod1_candidate_blocks`中的街区，随机种子固定为`20260920`。

## 核心样本

每个城市保留建筑对象数2至499的候选，按`lod1_building_count, block_id`稳定排序后切成4个等频组。建筑数并列值允许跨组，这是为了遵守“排序后的等频组”并保证每组规模。每组无放回抽10个，共24城×40=960个。随机数由全局种子、城市、组号的SHA-256派生，避免Python进程哈希差异。

如某组不足10个，先完成各组正常配额，再按组号距离由近到远、同距离取较小组号的顺序，从其他组未抽样街区中补足；补足来源写入`sampling_supplementation.csv`。本次实际数据无需补足。

## 压力与回归样本

- `single_building_stress`：每城从单建筑街区中确定性抽1个，共24个。
- `large_graph_stress`：每个存在不少于500栋街区的城市，取建筑数最接近500者；同差值依次按建筑数、`block_id`排序，共19个。
- `regression_anchor`：阶段2.5的15个固定案例。

同一`city + block_id`只编码一次，`sample_groups`保留全部组别。所有样本均标记为`development_audit`，不得作为后续未见测试集。

## 质量与冻结

抽样不因BCR异常、重复质心信号、指标缺失或后续构图结果替换案例。`quality_flags`只记录候选表当时可判定的异常；跨边界与cell失败留到编码阶段检查。完整219,419行抽样框（含未入选原因和26项指标）保存在`results/sampling_frame.csv.gz`，汇总在`results/sampling_frame_summary.csv`。

样本清单SHA-256记录在`results/database_snapshot.json`。后续编码只读该冻结清单，不重新抽样。
