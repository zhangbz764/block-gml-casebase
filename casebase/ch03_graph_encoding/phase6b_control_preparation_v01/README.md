# Phase 6B前置：形态控制目标与编码响应小试

任务ID：CH03-P6B-PREP-CONTROL-V01。2026年10月8日状态：**代码已编写，未运行，测试也未执行**。本轮直接要求优先于任务书的运行一次预览要求；没有Graph、数值响应、预览页面或研究结论。未commit。

按既有任务书直接实施，不另写长计划/台账，不启用子代理；测试断言先写，但不进行RED/GREEN运行，不能宣称测试通过。只有用户另行授权后才执行下列命令。

## 接口与来源

`control_probes.py`负责七个布局、小型本地适配器、响应摘要和限定真实案例读取；`render_control_probes.py`是本轮确有必要增加的绘图文件（统一平面范围、棱柱轴测及四部分短页面），复用6A的绘图/加载工具；`run_phase6b_preparation.py`仅组织本阶段输出。

现有编码接口可直接对本地Shapely几何调用，沿用v0.3.2和0.4m收缩/0.5m采样。建筑输入使用核心要求的`geometry/height/area/perimeter/building_id`；局部SRID 0满足旧接口的整数参数，但明确为未指定的本地米制坐标，不是假冒EPSG。编码核心自动填写的数据库来源，仅在新示意输出中纠正为synthetic_local；不改几何属性、边、标记、验证结果或旧代码。总体仅填面积、数量、BCR、面积加权平均高度，其余null及缺失原因。

真实案例只从当前6A `results/review/candidate_v01/pair_manifest.csv`读取；清单内容已与早先对话中的部分ID不同，因此不写死旧ID，运行时记录当前清单散列。按清单顺序，为每个目标取首个有两条相应有效边记录的案例，高度目标要求两条正高度差；这是可读记录的选择，不是自动辨认排列、组团或高层类型。三目标可复用同一街区；最多读60个已展示Graph，找到所需卡即停止；不足就注明，不重抽样或递归扫描。

沿用6A的Graph路径安全检查，以Phase 4B目录为基准。旧编码模块会导入psycopg2等原环境依赖，但只调用几何编码函数，**不调用数据库连接或全量运行入口**。不新装依赖，不重复搜索生成器，不复制真实Graph。生成器状态读取旧盘点；正式6B仍未就绪。

## 待授权的手动命令

在`E:\0_code\block-gml-casebase`运行：

```powershell
E:\0_python_envs\urbanCode\python.exe -B -m unittest discover -s casebase\ch03_graph_encoding\phase6b_control_preparation_v01\tests -p "test_*.py" -v
E:\0_python_envs\urbanCode\python.exe -B casebase\ch03_graph_encoding\phase6b_control_preparation_v01\run_phase6b_preparation.py --mode prepare --run-id preview_v01
```

没有默认运行模式，不可覆盖有内容的本阶段run目录；再运行需换run-id，不清空其他阶段。七个布局固定，无参数搜索；单布局编码失败保留失败行并继续指定剩余布局，最终返回非零。源文件缺失/越界或其他错误停止并留下简短失败metadata；不安装环境、不修复上游、不扩展实验。

先检查测试和返回码，再打开`results/review/preview_v01/visual_review.html`看比例、字体、图例及轴测。不要求漂亮响应或摘要全部单调；密集真实图的编号可结合端点表看。研究者只需对话回答想控制什么、先做哪个目标、哪些解释过度，无新增review CSV或输入表单。

## 输出与Git

本轮尚未创建results，以下只会在运行后产生：

- `results/probes/<run-id>/graphs/*.json`：七个小型示意Graph（失败项除外），保留端点与源记录；通用graphs忽略规则继续有效，仅交数据包，可重建。
- `results/probes/<run-id>/probe_response.csv`：一份紧凑表，图数量/面积/平均高度及相关边字段有效数量、中位数/IQR；null不填0，可进Git。
- `results/review/<run-id>/visual_review.html`与images/：仅四部分——间隔、错位、高度、至多三个真实卡；无表单，离线相对路径，数据包必须同时带图片。示意平面统一(-5,125)×(-5,85)米及8—24m色标；真实图用自己的有效高度范围、equal比例与源方向。弱边仍保留。
- `results/probes/<run-id>/input_manifest.json`：本轮配置/输入散列、来源、已读取/选定真实ID、输出状态、失败及绘图范围；首轮另存`results/input_manifest.json`，后续不覆盖首轮，可进Git。
- `results/control_preparation_decision_summary.md`、`results/phase6b_preparation_evidence_pack.md`：首轮短观察/建议及简报，Git忽略，另在对应probes/run目录保留同一说明。后续run只写自身目录。建议仍待研究者确认，不由程序判断形态正确性。

不生成bulk、dataset、pilot、全量张量、训练标签或日志台账；没有额外临时脚本/备份。少量图片如确有正文用途再明确选择，不默认上传全部图片。小Graph和图示虽可重建，审核工作区仍须得到完整数据包，不能假定Git包含它们。

## 研究边界

`control_targets_v01.md`明确三个目标的字段与操作范围，特别说明long_side_offset_norm不是本任务的横向错位量。旧6A分类建议原样保留，本阶段根据最新备忘改为显性控制：参考编码→目标关系/约束→可执行位置操作→编码核对；当前只准备局部响应小试，没有实现完整生成或转译。

运行后的建议暂优先讨论建筑间隔及所需接口，错位/高度的可操作映射待进一步确定；若没有可读响应，原样记录而非补样或放宽规则。缺接口写“可确定编码目标，生成实施待补接口”。无GNN优势、真实生成或design agency效果结论，交付后停止。
