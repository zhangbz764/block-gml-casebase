# Phase 4B：v0.3.2 全量构图准备与手动分批运行

本目录是 Phase 3.5B 冻结清单的正式构图入口。编码版本固定为 `0.3.2`，只新增建筑级几何容错，不改变既有 Tessellation 和图连接规则。

整理后，12份分批摘要合并为`results/audit/full_v032/part_summary.csv`与`status_by_city.csv`；原分批记录及部分准备态汇总保存在`results/records.zip`。正式Graph、冻结清单、代码散列和运行账本保留原路径。若需运行旧的part_06修复脚本，先按[统一整理说明](../README.md)恢复该阶段的records；准备态证据包合入`../evidence_pack.zip`。

## 冻结数据与分组

母清单包含 24 城、213,054 个街区，SHA-256 为 `fe98fadbe6880bdffe92bd3dcac250265aaefbb1c13d90a7f4698a384ce4e9e7`。Tokyo 按 `block_id` 升序后以 1-based 奇偶行拆分，其余城市不拆分。

| part | 内容 | 街区数 | pilot 预计小时 |
|---|---|---:|---:|
| `part_01` | Tokyo 奇数行 | 28,892 | 8.62 |
| `part_02` | Tokyo 偶数行 | 28,891 | 8.62 |
| `part_03` | Sapporo | 29,362 | 5.37 |
| `part_04` | Osaka | 23,073 | 6.67 |
| `part_05` | Chicago | 16,610 | 11.87 |
| `part_06` | Kanazawa、Lyon、Zurich | 12,620 | 3.68 |
| `part_07` | Toronto、Dresden、Bologna | 12,878 | 7.73 |
| `part_08` | Portland、Amsterdam | 12,085 | 3.93 |
| `part_09` | Berlin、Paris | 12,180 | 9.00 |
| `part_10` | Vienna、Boston、New York | 12,002 | 6.27 |
| `part_11` | Winnipeg、Washington D.C.、Marseille | 12,014 | 6.37 |
| `part_12` | Hamburg、San Francisco、Rotterdam、Luxembourg | 12,447 | 7.91 |

## 手动运行

必须由研究者每次明确指定一个 part 和对应 run ID；运行器没有 `--all`、默认 part 或自动继续下一 part 的功能。

```powershell
E:\0_python_envs\urbanCode\python.exe casebase\ch03_graph_encoding\phase4b_full_graph_batches_v032\run_phase4b.py --part-id part_01 --run-id full_v032_part_01 --chunk-size 100
```

运行其他分组时同时替换命令中的两处 `part_01`。启动前会重新核对 Phase 3.5B 母清单、Phase 4B 总清单、当前 part 清单和关键代码组合散列；已存在的其他 part 也必须使用同一代码组合。同一命令再次执行会跳过已有且一致的成功图，继续尚未处理的案例。失败案例默认保留；只有明确加入 `--retry-failed` 才会增加一次尝试：

```powershell
E:\0_python_envs\urbanCode\python.exe casebase\ch03_graph_encoding\phase4b_full_graph_batches_v032\run_phase4b.py --part-id part_01 --run-id full_v032_part_01 --chunk-size 100 --retry-failed
```

数据库连接沿用项目 `.env`，事务强制只读。数据库返回的建筑数与冻结清单不一致时，该案例直接记为 `failed`，不会生成 Graph。每个 part 正常结束后会自动生成紧凑摘要，不需要另跑分析命令；摘要中只要仍有 `failed` 或 `interrupted`，进程就返回非零状态。

## 输出与核验

- 图：`results/dataset/graphs/<part_id>/<city>/<chunk_id>/`
- 追加式案例账本：`results/runs/full_v032/<part_id>/encoding_status.csv`和`building_exclusions.csv`
- chunk结束时的原子快照：同目录下对应的`*_snapshot.csv`
- part 审计摘要：`results/audit/full_v032/<part_id>/`
- 冻结清单：`results/dataset/`（默认不进入 Git）
- 准备态摘要：`results/audit/prep/`

全部 12 份人工运行完毕后，只读核验命令为：

```powershell
E:\0_python_envs\urbanCode\python.exe casebase\ch03_graph_encoding\phase4b_full_graph_batches_v032\verify_phase4b.py --all-parts
```

若任何 part 缺失、状态未闭合、成功图缺失，或者Graph的v0.3.2版本、part、清单散列、代码组合散列和`validation.passed`不一致，命令会列明问题并以非零状态退出；它不会补跑或生成 Graph。全量核验通过前保留 Phase 4A pilot Graph。
