# 第三章阶段2运行说明

本目录保存编码规范v0.1与运行代码；运行产物统一写入`results/`。数据库操作只读，连接参数从仓库根目录`.env`读取。

整理后，历史核验记录在`results/records.zip`，22张旧图片在`results/history.zip`；样本清单、人工判断表和作为阶段2.5输入的15个Graph保留原路径。下文的历史输出路径可通过[统一整理说明](../README.md)中的恢复命令还原；证据包已合入`../evidence_pack.zip`。

## 运行

使用项目现有的`urbanCode` Python环境：

```powershell
E:\0_python_envs\urbanCode\python.exe casebase\ch03_graph_encoding\phase2_encoding_v01\run_phase2.py
```

运行器会：

1. 计算阶段1八个交付文件的SHA256；
2. 在PostgreSQL只读事务中提取15个街区；
3. 导出模型中立JSON图；
4. 重新读取JSON并执行一致性检查；
5. 生成15张四联核对图、6张高分辨率代表图和1张联系表；
6. 更新样本、验证、视觉判断和运行元数据。

运行器不创建schema或表，不写回数据库，不执行全量构图、GNN训练或邻接算法比较。

## 主要文件

- `encoding_spec_v0.1.md`：编码规则与边界；
- `encoding_fields_v0.1.csv`：逐字段定义；
- `results/sample_manifest.csv`：15个样本及选择依据；
- `results/graphs/*.graph.json`：可独立读取和重绘的样本图；
- `results/validation_summary.csv`：一致性检查与轻量数值诊断；
- `results/visual_judgement.csv`：执行工作区的建筑学初判，保留研究者终审列；
- `results/visual_checks/`：核对图与联系表；
- `results/phase2_evidence_pack.md`：阶段结论和交回说明。

## 测试

```powershell
E:\0_python_envs\urbanCode\python.exe -m unittest discover -s casebase\ch03_graph_encoding\phase2_encoding_v01\tests -v
```
