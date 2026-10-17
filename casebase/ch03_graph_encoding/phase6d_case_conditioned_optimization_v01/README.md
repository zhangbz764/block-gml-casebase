# Phase 6D：三档条件适应生成 v0.2

方法不变：原轮廓变换、固定节点和原参照边、三个关系项、相同Powell求解器。R加入关系项，B关闭关系项，其余相同；不训练、不用GPU，不改数据库或原Graph。

固定顺序：Bologna P4B_IT_BO_000304（13节点）、Boston P4B_US_BO_000046（9）、Berlin P4B_DE_BE_000370（15）。不再补采或选备选；三例组织说明尚待研究者通过对话提供，三档使用同一说明，不自动识别行列／围合／组团。

| 档位 | 边界长方向缩窄 | 等效FAR提高 | 目标总楼面面积提高 | 配置／run-id |
| --- | --- | --- | --- | --- |
| light | 5% | 10% | 4.5% | config_light_v02.json／adapt_v02_light |
| medium | 15% | 25% | 6.25% | config_medium_v02.json／adapt_v02_medium |
| strong | 30% | 60% | 12% | config_strong_v02.json／adapt_v02_strong |

楼面量按实际原／新场地面积重算，不等于FAR增幅。原P90限高、3m层高假定、连续高度、变量范围、关系容差／权重、可行性、6000计算／600秒上限保持不变。每档3×2×5=30次，总90次上限约15小时加准备和绘图，不是耗时承诺。三档同时改变边界和FAR，只讨论组合压力。

## 输入预览

本轮授权运行三个prepare，不授权正式求解。PowerShell：

```powershell
Set-Location E:\0_code\block-gml-casebase\casebase\ch03_graph_encoding\phase6d_case_conditioned_optimization_v01
$adaptPython = 'E:\0_python_envs\ch03Recon\python.exe'
& $adaptPython -X utf8 prepare_experiment.py --config config_light_v02.json --run-id adapt_v02_light
& $adaptPython -X utf8 prepare_experiment.py --config config_medium_v02.json --run-id adapt_v02_medium
& $adaptPython -X utf8 prepare_experiment.py --config config_strong_v02.json --run-id adapt_v02_strong
```

预览分别在results/review/adapt_v02_<level>/input_preview.html，总览在results/review/levels_v02/levels_overview.html。保存全部五个起点和诊断，图片只展示其中一个起点，避免冗余图集。同档R/B共用数组，不同档按各自新边界生成，不借用另一档米制中心。

看预览后请通过对话给三例各一句希望延续的特点，确认三档条件及警告；我会提供带完整ID和--description的三条可复制确认命令，不要求你整理ID或填写表格。基础接口是prepare_experiment.py --run-id adapt_v02_light --confirm-inputs，说明为空时追加三次--description 'CASE_ID=说明'。一档确认不解锁其余档；首档冻结后的描述必须与其他档一致，不随结果改写。容量冲突／未扰动初始方案已可行会阻止确认，先讨论，不自动降低目标或换案例。

## 研究者手动求解及评价

只有输入描述和条件确认冻结后才能求解，可先做轻档再讨论中／强档：

```powershell
& $adaptPython -X utf8 run_generation.py --run-id adapt_v02_light --case all --mode both
& $adaptPython -X utf8 evaluate_generation.py --run-id adapt_v02_light --render-review
& $adaptPython -X utf8 summarize_levels.py --run-ids adapt_v02_light adapt_v02_medium adapt_v02_strong --render-review
```

其他档只换run-id；可--case CASE_ID --mode both按案例运行。明确恢复才加--resume，跳过同设置已完成尝试，失败／超时不静默重试。相同run-id不同输入／配置／代码或合同禁止覆盖，改变须另用run-id；不补签旧校验。跨档表保留全部配对、失败、部分完成、未运行；未测可行数为null／空，不补0。异常／超时可能与失败交叉计数，不相加当总数。

## 输出和历史

- results/runs/<run_id>/：小型manifest、freeze、environment、metrics_summary.csv、evaluation_summary.json，可入Git；同目录phase6d_evidence_pack.md已忽略。
- results/bulk/<run_id>/：权威manifest／freeze、原形输入、五份共同起点、全部布局、完整差值／诊断、review_images，已有bulk规则忽略。不是重新编码的v0.3.2 Graph。
- results/review/<run_id>/及results/review/levels_v02/：只读HTML，已忽略，以相对路径引用bulk图。交接须连同这些被忽略的图和数据，不只交HTML或Git。
- results/levels_summary.csv：每案例×档位×版本共18行紧凑表，可入Git；总览链接单档图，不生成跨档重复图集。
- results/plan_review.md：只追加修订说明，已忽略。本轮检查摘要在results/runs/revision_v02/verification_summary.json，不覆盖旧根记录。

config_v01.json、adapt_v01、smoke_v01及根results原记录保持原样，属于历史未确认预览；代码修订后旧run校验失配属预期，不能从它继续求解。layout_solver.py不改。

修订测试：& $adaptPython -X utf8 -m unittest discover -s tests -p test_levels_v02.py -v。仅合成输入和results/bulk/unit_levels_*临时目录，自动清理，不调用真实求解、不重跑smoke_v01。JSON.tmp正常原子重命名，__pycache__可再生成；没有项目外临时代码／备份或环境安装，不commit／push。
