# 第三章阶段 3.5B：最终筛选冻结与图编码入口清单

本阶段只读取阶段 3.5A 已生成的全量压缩台账，冻结最终排除规则 v0.2，并生成后续图编码所需的唯一入口清单。本阶段不连接数据库、不读取几何、不构图。

## 冻结输入

- 输入：`../phase3_5a_block_screening_v01/results/bulk/full_v01/block_screening_ledger.csv.gz`
- 输入规模：24 城、219,419 个唯一街区
- 输入 SHA256：`e2eb993b7bf1b096e0e94dd42976d05046c7d72ea69be512581a63985c44cb9a`
- 阶段 3.5A 规则版本：`0.1`

## 最终规则 v0.2

- H01–H05 任一命中：排除
- S01、S03、S05：排除
- S02：仅保留标记
- 原 S04：仅保留标记；改用硬规则合格样本的建筑数全局 P99.5 作为规模排除阈值
- P99.5 阈值：`lod1_building_count >= 369` 时排除
- S06：写入组合原因，不产生额外排除数

主原因优先级依次为 hard 主原因、S01、S03、S05、P99.5。

## 执行与复核

```powershell
E:\0_python_envs\urbanCode\python.exe casebase\ch03_graph_encoding\phase3_5b_final_sample_freeze_v01\freeze_final_sample.py --source-run-id full_v01 --run-id final_v01
E:\0_python_envs\urbanCode\python.exe casebase\ch03_graph_encoding\phase3_5b_final_sample_freeze_v01\verify_final_sample.py --source-run-id full_v01 --run-id final_v01
E:\0_python_envs\urbanCode\python.exe -m unittest discover -s casebase\ch03_graph_encoding\phase3_5b_final_sample_freeze_v01\tests -v
```

脚本默认拒绝覆盖同名输出。需要重跑时，应先明确处理既有 `final_v01` 结果，再使用相同命令；禁止把旧结果静默覆盖。

## 输出

可重新生成、默认不进入 Git 的大文件：

- `results/bulk/final_v01/final_screening_ledger.csv.gz`
- `results/bulk/final_v01/excluded_blocks.csv.gz`
- `results/dataset/graph_encoding_manifest.csv.gz`

进入 Git 的紧凑审计与冻结信息：

- `results/audit/final_v01/final_filter_flow.csv`
- `results/audit/final_v01/final_filter_reason_summary.csv`
- `results/audit/final_v01/final_filter_overlap_summary.csv`
- `results/audit/final_v01/final_summary_by_city.csv`
- `results/audit/final_v01/graph_scale_summary.csv`
- `results/audit/final_v01/verification_summary.json`
- `results/input_snapshot.json`
- `results/freeze.json`
- `results/environment.json`

`results/phase3_5b_evidence_pack.md` 与 `results/run_log.txt` 是本地交接材料，按项目现有规则忽略。

## 阶段边界

本阶段止于最终样本和图编码入口清单冻结。后续图编码必须以 `graph_encoding_manifest.csv.gz` 为唯一入口；本阶段不启动全量图构建。
