# 第三章阶段3.5A：全量候选街区筛选与台账

本阶段为`summary.lod1_candidate_blocks`冻结可解释的街区级筛选规则、确定性pilot和正式全量运行入口。执行工作区只运行测试与pilot；219,419行正式全量台账由研究者审查代码后手动生成。

## 约束

- 实验ID：`CH3-P3.5A-BLOCK-SCREEN-V01`
- 规则版本：`0.1`
- 冻结母样本：24城、219,419个街区
- 数据库连接：`../shared/db_readonly.py`与项目根目录`.env`，只读事务
- 不访问全量建筑表，不执行建筑buffer、Tessellation、图构建或模型训练
- 三态：`exclude_hard`、`review_scope`、`keep_provisional`
- soft tail只触发复核，不自动排除

## 本轮执行

```powershell
E:\0_python_envs\urbanCode\python.exe casebase\ch03_graph_encoding\phase3_5a_block_screening_v01\run_phase3_5a.py --mode pilot --run-id pilot_v01
E:\0_python_envs\urbanCode\python.exe casebase\ch03_graph_encoding\phase3_5a_block_screening_v01\verify_screening.py --mode pilot --run-id pilot_v01
```

正式全量命令写在本README末尾及任务书中，本轮不会执行。

## 正式运行（研究者手动）

```powershell
E:\0_python_envs\urbanCode\python.exe casebase\ch03_graph_encoding\phase3_5a_block_screening_v01\run_phase3_5a.py --mode full --run-id full_v01 --expected-rows 219419 --expected-cities 24
E:\0_python_envs\urbanCode\python.exe casebase\ch03_graph_encoding\phase3_5a_block_screening_v01\verify_screening.py --mode full --run-id full_v01
E:\0_python_envs\urbanCode\python.exe casebase\ch03_graph_encoding\phase3_5a_block_screening_v01\build_review_sample.py --run-id full_v01 --max-cases 72
E:\0_python_envs\urbanCode\python.exe casebase\ch03_graph_encoding\phase3_5a_block_screening_v01\render_review_cases.py --run-id full_v01
```

预期完整台账路径：`results/bulk/full_v01/block_screening_ledger.csv.gz`；紧凑审计结果路径：`results/audit/full_v01/`；有限可视化路径：`results/visual_review/full_v01/`。

## S02/S04 单独命中补充复核

以下命令从既有 `full_v01` 台账确定性抽取 S02-only 与 S04-only 各 12 个案例，不重跑筛选规则；绘图步骤只查询入选的 24 个街区及其建筑几何。

```powershell
E:\0_python_envs\urbanCode\python.exe casebase\ch03_graph_encoding\phase3_5a_block_screening_v01\build_solo_flag_review.py --source-run-id full_v01 --output-run-id full_v01_solo --per-rule 12
E:\0_python_envs\urbanCode\python.exe casebase\ch03_graph_encoding\phase3_5a_block_screening_v01\render_review_cases.py --run-id full_v01_solo
```

可提交的小型结果位于 `results/audit/full_v01_solo/`；可重新生成且默认忽略的联系表与交互式页面位于 `results/visual_review/full_v01_solo/`。
