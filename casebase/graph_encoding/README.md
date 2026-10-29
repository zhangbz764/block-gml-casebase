# Graph encoding：正式街区图数据与编码

本目录只负责数据筛选、只读接入、0.3.2图编码和轻量读取，不包含生成模型、训练或形态研究实验。正式图保持原压缩JSON字节、原sample_id及part/城市/chunk分区，不因目录整理重新生成。

## 数量与边界

母样本219,419；最终编码入口213,054；有效Graph **213,052**；另2条为excluded_no_valid_building，不存在图文件。24城，共7,107,740节点、16,100,006边。不能把入口行数当作Graph文件数，城市来源表中的其他历史统计口径也不能替代此数量。

数据版本、相对路径、索引SHA和原始冻结依据见dataset.json；103字段定义见specs/feature_registry_v03.tsv。最终剔除H01—H05、S01/S03/S05及建筑数上尾p99.5（369及以上）；S02/S04仅保留标记，S06保留组合原因。具体公式/优先级以specs中的冻结规则和源码为准。

## 读取正式数据：只需Python标准库

在仓库根目录：
```python
from casebase.graph_encoding import GraphDataset

dataset = GraphDataset("casebase/graph_encoding/dataset.json")
record = dataset.get("P4B_JP_TK_035143")
nodes = record["nodes"]
edges = record["edges"]
boundary_wkt = record["graph"]["block_geometry_wkt_metric"]
```

GraphDataset以描述文件为路径基准，与当前工作目录无关。首次get加载索引的ID/状态/路径查找缓存，不加载全库Graph；每次仅解压所需案例。未知ID抛KeyError，排除ID抛ValueError，不回退SQL、不自动构图。索引原状态列名case_status保留不改名，graph_path相对results/dataset/。

不同城市的metric_srid可能不同；WKT使用各自米制投影，不能直接跨城市叠加。几何和高度字段按原编码返回，缺失值/缺失原因不擅自填补。来源依据和历史hash是原数据的记录，不等于整理后代码hash。

仓库外可用importlib从dataset_io.py直接加载，不必安装本项目；示例在临时验证目录E:/0_code/block-proj-split-test/。复制/迁移数据时带上本目录描述文件、读取器及其相对数据目录，不需要旧phase目录。

## 正式功能及环境

- core/building_features.py、assembly.py：建筑特征与基础组装。
- core/tessellation.py：原shrink/采样/cell/邻接算法，不改变语义。
- core/export_v03.py、building_filter.py、encoding_v032.py：分层属性、弱边标记、建筑排除及0.3.2导出。排除仅限polygonal extraction无正面积和shrink后为空，单节点照常保留。
- pipeline/：最终筛选、分组清单、只读批量接入、追加账本/原子快照与元数据核验。数据库仅在明确运行数据库命令时连接，沿用仓库根.env，不打印凭据。
- specs/：最终规则/参数、103字段、必要底层规范。旧v0.2/v0.3/0.3.1说明只解释继承部分，最终以0.3.2规格及当前正式源码为准。

编码/筛选复用E:/0_python_envs/urbanCode/python.exe及既有NumPy、SciPy、Shapely、pandas、psycopg2、python-dotenv；读取器本身不需要这些库，也不需要Torch。本次不新建环境或安装依赖。

## 显式手动入口（以下不是日常读取步骤）

从仓库根运行帮助不会构图/连接数据库：
```powershell
& 'E:/0_python_envs/urbanCode/python.exe' -m casebase.graph_encoding.pipeline.screening --help
& 'E:/0_python_envs/urbanCode/python.exe' -m casebase.graph_encoding.pipeline.final_screening --help
& 'E:/0_python_envs/urbanCode/python.exe' -m casebase.graph_encoding.pipeline.build_manifests --help
& 'E:/0_python_envs/urbanCode/python.exe' -m casebase.graph_encoding.pipeline.prepare_run --help
& 'E:/0_python_envs/urbanCode/python.exe' -m casebase.graph_encoding.pipeline.run_batches --help
& 'E:/0_python_envs/urbanCode/python.exe' -m casebase.graph_encoding.pipeline.verify_batches --help
```

筛选入口接受候选CSV（--input）或明确--database，只写新的--output目录；最终筛选入口读取保留的初筛ledger并核对其冻结SHA和正式数量。build_manifests保留原12份分组逻辑，需指定新的--output，不覆盖原清单。

确有重新编码需求时，先冻结整理后实现，再手动运行某part。例如：
```powershell
& 'E:/0_python_envs/urbanCode/python.exe' -m casebase.graph_encoding.pipeline.prepare_run --run-id replay_v032
& 'E:/0_python_envs/urbanCode/python.exe' -m casebase.graph_encoding.pipeline.run_batches --part-id part_01 --run-id replay_v032
& 'E:/0_python_envs/urbanCode/python.exe' -m casebase.graph_encoding.pipeline.verify_batches --part-id part_01 --run-id replay_v032
```

这些命令会产生新的编码运行；默认chunk100、0.4m shrink、0.5m边界采样及原弱边规则不变。运行前核对母/总/part清单SHA；新run的所有part使用同一新代码散列，不能拿原数据的旧散列冒充新实现。数据库建筑数量不匹配记failed且不生成图；failed/interrupted返回非零；断点核对已有图的元数据。

**新运行只写results/runs/<run_id>/，不会覆盖results/dataset/graphs/正式图。** 历史最终快照在results/metadata/encoding/runs/，不是可直接续写的活动运行目录。新输出如需成为新正式数据，须另外明确发布/替换计划，不在本整理任务内自动发生。

## 文件与Git

源代码、dataset.json、specs和少量合成测试进Git；整个results/及缓存默认忽略：

- results/dataset/：唯一正式Graph及新路径索引。
- results/metadata/：必要来源、筛选/编码冻结、原索引/统计，以及一份历史2007例成员/分区/曝光记录。
- results/cleanup/：一次确认清单和完成摘要。
- results/runs/：将来显式启动的新筛选/编码运行。

historical research资料不是正式全库训练划分；旧cache/graph_path只作历史追溯，新研究按sample_id查当前GraphDataset。旧研究源码/权重/缓存/页面已按确认清单移出当前数据项目，不保留兼容软链接或整体备份ZIP。已提交内容可查Git历史，原未跟踪/忽略的实验内容不能依赖Git恢复。

轻量回归：
```powershell
& 'E:/0_python_envs/urbanCode/python.exe' -B -m unittest discover -s casebase/graph_encoding/tests -p 'test_*.py'
```

回归使用极小离线输入：与整理前输出比较实际节点/cell/边/属性，验证最终筛选、追加账本和安全读取；不是全库重编码或完整数据质量重审。
