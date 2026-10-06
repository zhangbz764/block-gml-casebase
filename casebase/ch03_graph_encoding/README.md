# 第三章实验目录

各阶段保留独立代码、字段/参数定义和 README；源代码与测试不因结果整理而删除。

## 结果目录名称

- `audit`：运行后的统计、质量检查及资源估算结果。
- `pilot`：少量有代表性的案例试跑，用于试验方法和估算正式运行资源。
- `smoke`：更小规模的快速试跑，用于检查连接、读取、构图、保存的基本流程。
- `dataset`：正式数据、入口清单和必要的索引。
- `review` / `visual_review` / `visual_checks`：人工查看所需的清单、图表或页面。

## 整理后的保存规则

代码、配置、规范以及有研究价值的精简统计可以进入 Git，不实行“全部 results 都忽略”。

- `results/records.zip`：已完成阶段的历史 CSV/JSON 记录，保留原目录和原始字节；普通 ZIP 软件即可查看。作为精简阶段档案保留在 Git。
- `results/history.zip`：旧试跑图、可重绘图片、Cell 副本等生成物，合并保存，Git 忽略。
- 本目录的 `evidence_pack.zip`：证据包、计划审查等交接材料，按原阶段路径合并保存，Git 忽略。研究者当前需要阅读的 Phase 5 `researcher_summary.md`仍单独保留。
- 后续代码仍会读取的清单、样本图、人工判断表，以及 Phase 4B/5 冻结数据、账本和索引，保持原路径。

Phase 4B 的12份分批摘要另合并为 `phase4b_full_graph_batches_v032/results/audit/full_v032/part_summary.csv` 和 `status_by_city.csv`，日常阅读这两张表即可。原分批 JSON/CSV（包括空失败表）保留在该阶段的 `records.zip`。

## 复跑已归档的旧阶段

旧脚本仍按原路径读取，不需要修改已经冻结的编码代码。复跑旧报告或核验前，先恢复相应阶段：

```powershell
E:\0_python_envs\urbanCode\python.exe casebase\ch03_graph_encoding\shared\compact_results.py --restore phase4a_chunked_graph_pilot_v01
```

恢复保持文件字节不变，遇到同名但内容不同的文件会停止，避免覆盖后续修改。`--bundle records`可只恢复轻量记录，`--bundle handoff`可只恢复该阶段的交接文档；`--restore all`可恢复全部历史阶段。复跑 Phase 2 的原始冻结流程时，还应恢复 `phase1_audit` 的交接文档。

整理命令不连接数据库、不构图。默认只显示范围；`--apply`将明确列出的历史文件合并归档，然后删除散文件，保留当前正式运行资产：

```powershell
E:\0_python_envs\urbanCode\python.exe casebase\ch03_graph_encoding\shared\compact_results.py
E:\0_python_envs\urbanCode\python.exe casebase\ch03_graph_encoding\shared\compact_results.py --apply
```

`records.zip`、`history.zip`、`evidence_pack.zip`是正式保留的整理产物，不是临时备份。ZIP 内的 `__manifest__.json`包含每个文件的 SHA-256；删原文件前先校验归档内容。重新运行生成的同名文件若与归档内容不同，需作为新一轮结果单独保存，整理器不会覆盖历史版本。

今后的新阶段优先只生成必要输入、一个运行/核验摘要、必要统计表和人工判断。只有需要断点恢复或分批计算时才保留逐批账本；交接说明与开发检查材料按上述忽略规则管理，避免不断增加零散输出。
