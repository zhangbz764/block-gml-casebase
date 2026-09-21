# 第三章阶段3.1：异常建筑排除与编码收尾

本阶段在阶段3冻结样本上执行一次局部编码修订：固定使用`shrink_m=0.4`，将内缩后无有效面状几何的建筑作为`empty_after_shrink`异常对象，在node派生前排除。数据库连接保持只读。

阶段3基线不被覆盖。正式重建范围仅为旧结果中`cell_failure_count > 0`的57个街区；另外3个无失败案例只做临时回归签名比较。阶段3其余946个成功图和14个`Invalid footprint`失败街区保持原状态。

## 冻结输入

- 样本清单：`../phase3_encoding_dataset_v03/results/sample_manifest.csv`
- 样本清单SHA-256：`8679cfcd4a95335f0b8ff4359bceef8df0d97d85c4683fc1a47f86e59f1e86bb`
- 数据库快照：`../phase3_encoding_dataset_v03/results/database_snapshot.json`
- 旧状态表：`../phase3_encoding_dataset_v03/results/audit/encoding_status.csv`
- 旧图目录：`../phase3_encoding_dataset_v03/results/dataset/graphs/`
- 受影响清单：`results/affected_manifest.csv`（57例）
- 回归清单：`results/regression_manifest.csv`（3例）

只读核对确认：57个受影响街区的旧结果含308个`empty_after_shrink`记录；`p3__CH_ZU_000062`是唯一原节点全部失败的案例。

## 执行

```powershell
E:\0_python_envs\urbanCode\python.exe casebase\ch03_graph_encoding\phase3_1_encoding_cleanup_v031\run_phase3_1.py --scope affected
E:\0_python_envs\urbanCode\python.exe casebase\ch03_graph_encoding\phase3_1_encoding_cleanup_v031\verify_phase3_1.py
```

运行产物统一写入`results/`。其中`results/dataset/graphs/`、运行日志、证据包和计划审查文件按仓库通用规则不进入Git。
