# Phase 5：全量图数据集审计与编码资产冻结

当前状态：**PLANNED / CODE_READY**。

本目录已经准备好只读审计代码、合成测试、审计规范和103项字段注册表快照。Phase 4B仍在手动运行，Phase 5正式运行尚未开始；当前目录中的准备态材料不包含任何正式全量Graph统计，也不能表述为`EXECUTED`、`VERIFIED`或`CLAIM_READY`。

## 正式运行前提与顺序

必须严格按以下顺序进行：

1. 完成Phase 4B全部12个part；
2. 捕获Phase 4B全量核验；
3. 运行Phase 5全量只读审计；
4. 检查自动核验摘要；
5. 由研究者查看48例可视化HTML；
6. 最后再形成阶段结论。

Phase 5不连接数据库，不重算Tessellation，不修改Graph，也不新增筛选规则。`excluded_no_valid_building`仍保留为全量索引中的合法入口状态。

## 正式命令（现在不要运行）

在项目根目录依次执行：

```powershell
E:\0_python_envs\urbanCode\python.exe casebase\ch03_graph_encoding\phase5_graph_dataset_audit_v01\phase4b_gate.py --capture
E:\0_python_envs\urbanCode\python.exe casebase\ch03_graph_encoding\phase5_graph_dataset_audit_v01\run_phase5.py --all-parts --run-id full_v01
E:\0_python_envs\urbanCode\python.exe casebase\ch03_graph_encoding\phase5_graph_dataset_audit_v01\verify_phase5.py --automated
```

如需分part建立审计分片，可显式运行：

```powershell
E:\0_python_envs\urbanCode\python.exe casebase\ch03_graph_encoding\phase5_graph_dataset_audit_v01\run_phase5.py --part-id part_01 --run-id full_v01
```

运行器没有默认正式模式；Phase 4B门槛、输入快照或既有分片散列不一致时会停止。全量模式逐part流式读取，每次只保留一个Graph与紧凑累计量；不生成全量Node/Edge平铺表。

## 输出与恢复

- `results/dataset/index_parts/`：12个可恢复索引分片；
- `results/dataset/graph_dataset_index.csv.gz`：213,054行冻结入口索引；
- `results/audit/full_v01/`：总体、城市、结构分位数、字段完整性、问题与自动核验摘要；
- `results/review/full_v01/`：48例清单、图片、离线HTML和人工复核空表；
- `results/input_snapshot.json`、`environment.json`、`freeze.json`：输入、环境与最终散列。

已有part分片只有在manifest、状态账本、审计规范、Phase 4B与Phase 5代码组合散列、行数和输出散列全部一致时才会跳过；审计代码变化后旧分片会被明确拒绝，不一致时停止且不覆盖。

正式运行完成后，研究者优先阅读`results/audit/full_v01/researcher_summary.md`，再打开`results/review/full_v01/visual_review.html`查看48例。人工复核不要求理解内部散列、分片签名或逐行核验字段，也不要求逐行填写48例：若总体正常，只需形成一段总判断；只有发现问题时才需要在`manual_review.csv`记录对应案例。最终只需回答摘要列出的四个研究判断问题。

## 开发测试

本机环境当前没有安装`pytest`，因此准备阶段使用标准库`unittest`运行同一组测试，无需安装额外依赖：

```powershell
E:\0_python_envs\urbanCode\python.exe -m unittest discover -s casebase\ch03_graph_encoding\phase5_graph_dataset_audit_v01\tests -p "test_*.py" -v
```

测试只使用临时目录中的合成Graph，不读取正在写入的Phase 4B正式part。若以后安装了`pytest`，这些`unittest`测试也可由pytest直接收集。

## Git范围

源代码、测试、README、审计规范、注册表快照和`results/audit/prep/verification_summary.json`适合进入Git。全量索引、part中间分片、图片、HTML、pilot、运行日志、证据包和内部判断文档由`.gitignore`排除；正式完成后的紧凑总体审计摘要可单独纳入版本记录。
