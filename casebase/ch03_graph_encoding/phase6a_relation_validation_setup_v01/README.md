# Phase 6A：形态对照案例与模型验证准备

2026年10月6日，经研究者授权已运行2对真实预览与正式候选页，未训练模型。只读Phase 5索引与入选v0.3.2 Graph，不连接数据库、不改写freeze、不重算Cell、不遍历全库Graph。

整理后的项目输入路径与任务书相符，无须解开历史归档。正式开发池23,456例、覆盖24城；入选30对，全部为close，按需读取60个Graph并输出150张图片。人工判断尚未提供，不能由运行成功推定案例有解释价值。生成器接口在当前项目未找到，`results/task_definition_v01.md`中暂记`ready_for_phase6b=false`；不影响候选页面。

真实运行时修复了入口检查：两个合法的`excluded_no_valid_building`入口允许缺少Graph路径及国家字段；completed入口仍要求这些字段完整。未改变上游文件、抽样/配对规则或输入语义。15项小测试通过。

## 手动运行

首次运行使用的命令如下（已完成，现有run-id不可覆盖；需要新一轮须换新run-id）。在`E:\0_code\block-gml-casebase`打开终端：

```powershell
E:\0_python_envs\urbanCode\python.exe casebase\ch03_graph_encoding\phase6a_relation_validation_setup_v01\run_phase6a.py --mode smoke --run-id smoke_v01 --max-pairs 2
E:\0_python_envs\urbanCode\python.exe casebase\ch03_graph_encoding\phase6a_relation_validation_setup_v01\run_phase6a.py --mode candidates --run-id candidate_v01
```

分别打开`results/review/smoke_v01/visual_review.html`和`results/review/candidate_v01/visual_review.html`。每对有总体指标、同尺度/共同高度色标的平面图、普通/弱边及Cell切换、关系分布简表和最多6条编号示例边。页面离线使用，转交需带上images/，不能只发HTML。

smoke每城至多100例、最多2对；正式每城至多1,000例、最多30对。SHA-256抽样、同城median/IQR距离、等级优先与轮转上限见`pairing_spec_v01.json`；不足时如实报告，不插补或循环放宽。nearest_only只补到20对，且不能直接视为指标接近。无边Graph及少量缺失Cell继续保留。

有内容的run目录拒绝覆盖，勿清空已有人工判断来重跑。需要新一轮时使用新run-id；首轮正式开发清单固定，复用时必须一致。命令失败返回非零；如在图示制作中失败，保留该run的简短verification_summary，修复后使用新run-id。输入口径或SHA不符则停止，不修复上游。没有候选也交付空表、说明和页面，不伪造成功案例。

## 只保留必要记录

根目录为源代码、测试、参数和人工输入合同。`results/`为输出：

- `sampling/development_manifest.csv`：正式开发池，可重建，Git忽略；smoke开发池只在自己的run目录。
- `review/<run-id>/pair_manifest.csv`、`pair_metrics.csv`、`relation_summary.csv`、`example_edges.csv`：紧凑结果，可进Git；不是全城两两距离表。
- `review/<run-id>/images/`、HTML、smoke目录：可重建，Git忽略；审核数据包仍需包含相对路径与图片。
- `review/<run-id>/pair_review.csv`：人工空表，不代填“通过”；暂忽略，填写后可用`git add -f`只加入这一份人工记录。页面输入不自动保存，点“导出人工判断CSV”后把下载文件放回对应run目录。
- `input_manifest.json`、`sampling_summary.json`、`verification_summary.json`：首个成功正式run的紧凑记录，可进Git；每个run同时保留自身记录，后续run不覆盖首轮根目录记录。
- `generator_interface_inventory.csv`、`task_definition_v01.md`：本轮人工盘点与任务定义，现已交付，可进Git。运行命令不会自动修改接口盘点。
- `phase6a_evidence_pack.md`：运行后1—2页简报，Git忽略；首个成功正式run在results根目录另存同一简报，不另立多份审查文档。

暂未设置训练/测试划分；开发池已查看，后续不可称未见测试集。关系摘要不是latent space或额外模型特征，程序通过不等于形态判断通过。

## 输入与研究判断

输入合同见`model_input_contract_v01.md`及`model_input_fields_v01.csv`；原15/8/12核心字段不变。仅选中Graph做统一字段提取检查，不输出全量模型文件。节点排序及端点重映射一致，双向边交换i/j及掩码，缺失高度保留null而非0。

研究者可只选约8—12对并给一段总判断：指标是否足够接近、空间组织差异是什么、连接和属性是否帮助表达差异。没有差异同样可记录。不自动生成类型标签，也不从可视化正常推出图模型优越。

## 小测试

```powershell
E:\0_python_envs\urbanCode\python.exe -B -m unittest discover -s casebase\ch03_graph_encoding\phase6a_relation_validation_setup_v01\tests -p "test_*.py" -v
```

只用合成数据验证固定抽样/配对、缺失与常量处理、Graph路径、共同输入与体量、端点映射、无边/无Cell、绘图比例和输出保护。临时测试目录由标准库自动删除，不生成真实研究结果，也不重测历史构图算法。
