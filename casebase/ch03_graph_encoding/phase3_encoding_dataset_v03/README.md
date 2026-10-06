# 第三章阶段3：中等样本扩展与编码定型

本目录从只读数据库和阶段2/2.5冻结编码生成阶段3开发与审计数据，所有运行产物统一写入`results/`。数据库不写入，不建立新schema；阶段1、2和2.5产物不覆盖。

整理后，历史抽样/审计/预运行记录在`results/records.zip`，旧图片等生成物在`results/history.zip`。后续阶段使用的样本清单、数据库快照、字段注册表、正式状态账本和图数据仍保留原路径。复跑历史报告前按[统一整理说明](../README.md)恢复记录；证据包和计划审查已合入`../evidence_pack.zip`。

固定实验ID：`CH3-P3-ENCODING-SCALE-V03`。

执行顺序为：阶段2.5收尾冻结、确定性抽样、约100例预运行、资源门槛检查、中等样本批处理、特征与图结构审计、有限可视化和证据包。阶段3不训练模型、不删除弱边、不扩展到全库。

确定性抽样已冻结为1,017个去重街区；`results/sample_manifest.csv`是后续批处理唯一清单。`results/sampling_frame.csv.gz`保存完整抽样框和未入选原因，属于正式追溯数据。

阶段3实际完成1,003个图，14个源建筑底面无效街区作为硬失败保留在`results/audit/encoding_status.csv`。完整结论、边界与待决事项见`results/phase3_evidence_pack.md`。

主要入口：

- `sample_phase3.py`：生成只读快照和固定样本；
- `run_phase3.py --mode pilot|full`：预运行或全量编码；
- `audit_phase3.py`：导出模型表并生成描述性审计；
- `visual_review_phase3.py`：生成有限联系表和分布图。

运行环境使用`E:\0_python_envs\urbanCode\python.exe`。图文件为确定性gzip JSON；节点、cell和边表为CSV或gzip CSV。
