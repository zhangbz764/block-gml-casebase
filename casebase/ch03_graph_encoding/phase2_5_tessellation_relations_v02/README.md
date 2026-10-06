# 第三章阶段2.5：形态Tessellation与关系属性v0.2

本目录只生成阶段2固定样本的派生数据，运行产物统一写入`results/`；不写回数据库，不修改或覆盖`phase2_encoding_v01/results/`中的v0.1输入。

整理后，历史核验/敏感性记录在`results/records.zip`；Cell副本、旧图片及页面在`results/history.zip`。输入/样本清单、15个Graph和人工判断表仍保留原路径。下文的历史输出路径可通过[统一整理说明](../README.md)中的恢复命令还原；证据包及开发检查文档已合入`../evidence_pack.zip`。

## 目标

在阶段2的15个固定样本上，从建筑轮廓生成Morphological Cell，以cell共享线状边界定义候选邻接，并保存距离、共享界面、朝向、侧边投影、尺度和高度关系属性。阶段2.5只验证该空间骨架是否可计算、可解释和可追溯，不训练GNN/GAT，也不扩大到全库。

## 固定参数

- 建筑轮廓向内收缩：`0.4 m`
- 主轮廓离散间距：`0.5 m`
- 敏感性间距：`1.0 m`、`2.0 m`
- 最小共享线状边界：`1e-6 m`

## 运行顺序

```powershell
E:\0_python_envs\urbanCode\python.exe -m unittest discover -s casebase\ch03_graph_encoding\phase2_encoding_v01\tests -v
E:\0_python_envs\urbanCode\python.exe -m unittest discover -s casebase\ch03_graph_encoding\phase2_5_tessellation_relations_v02\tests -v
E:\0_python_envs\urbanCode\python.exe casebase\ch03_graph_encoding\phase2_5_tessellation_relations_v02\run_phase2_5.py
E:\0_python_envs\urbanCode\python.exe casebase\ch03_graph_encoding\phase2_5_tessellation_relations_v02\visual_review.py
```

研究者判断字段保留在`results/visual_review_decisions.csv`中，不由脚本自动填写。

## 主要复核入口

- `results/phase2_5_evidence_pack.md`：阶段结果、限制和下一步决定清单。
- `results/visual_checks/contact_sheet.png`：15个样本四联联系表。
- `results/visual_checks/representative/`：5个代表样本六联关系诊断图。
- `results/visual_review.html`：无外部依赖的离线图层与边属性复核页。
- `results/visual_review_decisions.csv`：研究者判断表；人工字段为空。
- `results/validation_summary.csv` / `results/validation_issues.csv`：自动检查与可追溯异常。
- `results/parameter_sensitivity.csv`：4个样本、3种离散间距的有限敏感性结果。

如直接从文件管理器打开 `results/visual_review.html`，浏览器应在本地离线显示；不需要启动服务器。
