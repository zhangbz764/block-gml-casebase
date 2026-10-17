# Phase 6C：图编码与布局重构准备

用真实v0.3.2图压缩到32维，再在已知街区边界／建筑数下重构无序矩形柱体。A=边属性GAT，B=无边建筑集合MLP。定义见[experiment_contract_v01.md](experiment_contract_v01.md)，预算见[config_v01.json](config_v01.json)。不是新设计生成；一次首轮A/B完成后先讨论，不自动追加实验。

当前状态（2026-10-10）：独立运行环境已配置；11项检查全部通过，8例／4优化步GPU短检查通过。尚未正式准备、训练或评价，短检查不是效果证据。实况以`results/verification_summary.json`及`results/environment.json`为准。

## 环境与工作目录

本项目VS Code默认解释器仍为`E:\0_python_envs\urbanCode\python.exe`，用于原有阶段；没有覆盖或升级它。未激活conda的终端里，裸`python`可能指向Miniconda base，并不等于VS Code选择的环境。

6C使用已配置的`E:\0_python_envs\ch03Recon\python.exe`：Python3.12.13、torch2.9.1+cu128、PyG2.7.0、numpy2.3.5、scipy1.16.3、shapely2.1.2、matplotlib3.10.7。RTX4070 CUDA前向／反向短检查已通过，pip check无依赖冲突。系统驱动／CUDA安装及其他conda环境保持不变。不需要先运行conda activate，以下绝对解释器命令可直接使用。

主依赖版本和GPU包索引保留在`environment.yml`，可用于在其他机器重建独立环境（不对现有同名环境再次执行create）。安装依据：[PyTorch官方GPU包命令](https://pytorch.org/get-started/previous-versions/)、[PyG最小安装说明](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)。没有安装本任务不需要的torchvision、torchaudio或PyG编译扩展。匹配使用[SciPy官方接口](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html)。

以下在PowerShell运行；路径已确认存在：

```powershell
Set-Location E:\0_code\block-gml-casebase\casebase\ch03_graph_encoding\phase6c_graph_layout_reconstruction_v01
$reconPython = 'E:\0_python_envs\ch03Recon\python.exe'
# 可选复查；本轮已通过，不必为了正式运行重复执行。
& $reconPython -X utf8 short_check.py --config config_v01.json
```

## 四个正式手动命令（本轮没有执行）

环境及模型短检查已通过。研究者可按顺序手动运行，不并行抢显存：

```powershell
& $reconPython -X utf8 prepare_dataset.py --config config_v01.json --run-id recon_v01
& $reconPython -X utf8 train_reconstruction.py --config config_v01.json --run-id recon_v01 --model relation
& $reconPython -X utf8 train_reconstruction.py --config config_v01.json --run-id recon_v01 --model set
& $reconPython -X utf8 evaluate_reconstruction.py --run-id recon_v01 --models relation,set --render-review
```

明确恢复中断的同一模型时才追加`--resume`，不恢复别的模型，已完成运行拒绝重训。相同run-id必须相同配置、输入清单、合同和关键代码；如配置／代码调整，应保留旧记录后另建run-id。显存不足停止，先只考虑降低batch并重新确认固定配置，不自动减少样本／改模型／调参。正式CPU仅可人工明确改配置、另建运行并自行接受耗时。

准备命令只读Phase5索引／注册表和Phase4B Graph，不连接数据库。索引路径相对Phase4B解析，拒绝绝对路径及逃逸路径；几何目标只读取选中的图，不扫全库。冻结split后拟合train统计。训练与验证只读张量缓存，不读测试集来选模型；两个模型完成后才评价test。

## 输出与保留

所有产物进入`results/`。紧凑记录`sample_manifest.csv`、`manifest.json`、`freeze.json`、`environment.json`、`verification_summary.json`、`metrics_summary.csv`可入Git。根目录只保留源码、测试、规范、配置和README。

`results/bulk/recon_v01/`存派生张量、normalization.json、样本元信息、latest/best、预测和逐例评价，使用现有bulk忽略规则。manifest及freeze记录标准化和缓存SHA，不复制原Graph。`results/review/recon_v01/visual_review.html`默认12个测试案例，四栏原轮廓／矩形目标／A／B，同一尺度、色标、相机，2D/3D切换，无表单、无几何修复。HTML已忽略；少量图可选入Git。`results/bulk/handoff/`是两份任务原件的正式本地交接副本，不是临时备份；`results/plan_review.md`和证据包不进Git。

若中断，checkpoint原子替换可能残留`latest.pt.tmp`或`best.pt.tmp`，位于已忽略的bulk下；不是额外历史快照。当前实施没有生成checkpoint。Python可能产生已忽略`__pycache__`，可随时再生成。

开发检查：`& $reconPython -X utf8 -m unittest discover -s tests -v`。使用旧urbanCode时模型测试仍会skip，不能把skip视为模型通过。
