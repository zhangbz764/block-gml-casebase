# 第三章阶段 4A：分块构图预运行与资源估算

本阶段以 Phase 3.5B 冻结的 213,054 个街区入口清单为唯一选择来源，按城市和至多 100 个街区一批进行只读数据库查询，并直接复用 Phase 3.1 的 `encode_block_to_v031` 与固定 Tessellation 参数。本阶段不创建数据库对象、不修改编码语义、不训练模型，也不启动全量构图。

## 冻结输入与抽样

- 输入：`../phase3_5b_final_sample_freeze_v01/results/dataset/graph_encoding_manifest.csv.gz`
- SHA256：`fe98fadbe6880bdffe92bd3dcac250265aaefbb1c13d90a7f4698a384ce4e9e7`
- 输入规模：24 城、213,054 个街区
- pilot：每城内部按建筑数量划分 4 个等频层，每层确定性抽取 25 例，共 2,400 例
- smoke：每个城市×层取 1 例，共 96 例
- 固定种子：`20260922`

生成命令：

```powershell
E:\0_python_envs\urbanCode\python.exe casebase\ch03_graph_encoding\phase4a_chunked_graph_pilot_v01\build_pilot_manifest.py
```

## 96 例 smoke 结果

执行命令：

```powershell
E:\0_python_envs\urbanCode\python.exe casebase\ch03_graph_encoding\phase4a_chunked_graph_pilot_v01\run_phase4a.py --mode smoke --run-id smoke_v01 --chunk-size 100
E:\0_python_envs\urbanCode\python.exe casebase\ch03_graph_encoding\phase4a_chunked_graph_pilot_v01\analyze_phase4a.py --mode smoke --run-id smoke_v01
```

结果：

- 96/96 完成，覆盖 24 城和全部 96 个城市×建筑数量层
- 24 个城市批次全部查询成功，无街区缺失、建筑数量不一致或编码失败
- 实际墙钟时间约 114.9 秒；其中批量查询 8.45 秒、逐图编码合计 102.22 秒
- 96 个图共 6,424,740 字节
- 7 个街区中的 18 栋建筑按 v0.3.1 规则标记为 `empty_after_shrink`，均被显式记录；没有街区因此变成空图
- Python `tracemalloc` 记录的单图峰值为 2.936 MB；该指标不包含数据库服务端或 GEOS 原生内存

## 全量资源估算

smoke 每层只有 1 例，因此外推时在 pilot 权重 `stratum_population / 25` 基础上乘以 25。按 96 个城市×层分别加权后：

- 数据库查询时间：27,754.6 秒，约 7.71 小时
- 编码时间：201,150.2 秒，约 55.88 小时
- 总运行时间：228,904.8 秒，约 63.58 小时或 2.65 天连续运行
- 图文件数量：213,054
- 图文件总大小：13,533,625,759 字节，约 12.60 GiB
- 查询约占总时间 12.12%，主要耗时来自图编码

本次没有查询超时或城市集中失败，CSV 驱动的只读分块查询足够稳定；当前建议是不建立数据库辅助表。

## 2,400 例 pilot 补充结果

研究者随后手动完成了 `pilot_v01`：2,400 例全部进入账本，2,398 例完成，2 例因单栋建筑存在无有效面状面积而失败，分别为 Boston `US_BO_003310` 和 San Francisco `US_SF_000689`。没有查询超时或批次缺失；这两例成为后续 v0.3.2 建筑级容错的冻结回归案例。

pilot 加权估算为：数据库查询约 2.20 小时、编码约 83.83 小时、总计约 86.03 小时，图文件约 18.78 GiB。查询只占预计总时间约 2.55%，因此继续采用 CSV 驱动的只读分块查询，不建立数据库辅助表。

## 恢复与输出策略

- 图先写入临时压缩文件，通过读取校验后原子改名。
- `completed` 图只有在入口清单散列、编码版本、运行 ID 和参数散列一致时才会跳过。
- 失败案例默认不重试；显式加入 `--retry-failed` 后才增加 attempt。
- 每个案例立即刷新 `encoding_status.csv`，每个城市批次原子刷新 `chunk_status.csv`。
- `results/smoke/` 和 `results/pilot/` 是可重新生成的阶段数据包，不进入 Git；抽样清单、输入快照及 `results/audit/` 紧凑摘要进入 Git。

pilot 的手动运行命令为：

```powershell
E:\0_python_envs\urbanCode\python.exe casebase\ch03_graph_encoding\phase4a_chunked_graph_pilot_v01\run_phase4a.py --mode pilot --run-id pilot_v01 --chunk-size 100
E:\0_python_envs\urbanCode\python.exe casebase\ch03_graph_encoding\phase4a_chunked_graph_pilot_v01\analyze_phase4a.py --mode pilot --run-id pilot_v01
```

本阶段停止于 96 例 smoke 与资源估算，不代替研究者启动 2,400 例 pilot，也不执行 213,054 例全量构图。
