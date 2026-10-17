# Phase 6C 重构合同 v0.1

研究边界：v0.3.2建筑关系图或无边建筑集合压缩成32维z，在已知街区边界和建筑数量N下重构无序矩形柱体。不是新设计生成、异构图学习、图编码格式重构或“103项均为独立特征”。不修改数据库、原Graph、冻结清单或旧阶段。

## 输入与变换

模型A/B共用图级15项：`elongation compactness bcr convexity rectangularity far bcd osr avh_w hcv footprint_cv avg_complexity avg_rectangularity avg_convexity orientation_entropy`。保持注册表原变换，仅训练集均值填充和标准化，再拼接15个缺失掩码（true=missing）。

建筑级共用10项（原8项core加2项派生通道）：

| 字段 | 适配层变换 |
| --- | --- |
| footprint_compactness, footprint_rectangularity, footprint_convexity | 保持原值 |
| orientation_sin_2theta, orientation_cos_2theta | 保持原轴向表达 |
| local_x_normalized, local_y_normalized | 原约定为街区几何质心／sqrt(area)，本轮由未收缩建筑轮廓质心重新表达为bbox中心平移／L |
| boundary_distance_normalized | 由原boundary_distance_m／L重新表达 |
| footprint_area_m2 | footprint_area_m2／L²；不新增原Graph字段 |
| height_m | log1p(height_m) |

随后每列按训练集观测值均值填充、标准化，拼接10个缺失掩码，总20通道。全缺失列使用均值0、尺度1；恒定列尺度1。仅适配位置与距离单位，不改变原编码资产。合法几何已存在时，新坐标是可观测值；不继承原坐标约定产生的无意义缺失标记。

模型A专用边级12项：`shared_boundary_ratio_min shared_boundary_ratio_mean footprint_gap_block_norm centroid_distance_block_norm orientation_similarity long_side_overlap_ratio long_side_offset_norm connection_axis_alignment_i connection_axis_alignment_j log_area_ratio_abs log_height_ratio_abs volume_ratio`。保持原Graph值和原编码尺度，包括原block_norm定义；不是新bbox坐标位置通道。训练集填充、标准化＋12缺失掩码，总24通道。保留弱边；双向化先交换i/j属性，NaN随交换，随后填充并生成相应掩码。自环只在GAT内部产生，属性填0，不写Graph。

ID／城市／源索引／状态／质量标记不进入网络；building_id只映射边端点和检查重复。WKT只制作任务目标、边界条件及明确说明的位置坐标适配，不输入轮廓向量。L及平移量只用于数据适配、展示与米制评价，不另输入解码器。

## 目标、边界与输出

街区包围盒中心平移，L=max(width,height)，xy及尺寸统一除L。不分轴缩放，不旋转街区。边界只由原街区面（保留孔洞）在[-0.5,0.5]²生成32×32像素中心覆盖掩膜，不含建筑占据图。

目标来自未收缩建筑WKT的最小有向包围矩形：矩形中心xy、long≥short、sin2θ/cos2θ、log1p原高度。目标高度独立按训练集目标建筑log高度均值／标准差转换；解码时expm1回米。图级指标和建筑质心可包含任务强信息，A/B相同，不故意削弱集合对照。

GAT两层，每层2头×32=总隐藏64；mean pooling＋block30→z32。B为20→104→64逐节点MLP，采用同一mean pooling及投影，宽度104用于近似参数量对齐，实际数目由运行记录，不声称绝对同容量。两模型各自独立训练；先初始化相同解码器再初始化编码器，保证固定种子下初始解码器权重相同，不共享训练后的权重。

解码器仅接受`forward(z, boundary, counts)`：边界卷积→16维条件，N／64，64个可学习16维查询；前N个输出7项。long=softplus(raw_long)+short，short=softplus(raw_short)，均加小正数；方向归一化。高度输出为无界标准化log高度，不以裁剪／几何修复改善结果；若逆变换产生负高度，单独报告负高度比例并如实绘图；非有限输出直接失败。无节点隐向量、边、原坐标、完整指标、目标顺序跳接。

## 匹配、损失与评价

SciPy Hungarian匹配，仅用于损失和评价。SmoothL1 beta=1：位置两通道均值、尺寸两通道均值、高度单通道；各权重1。方向=(1−单位轴向向量点积)×0.1×clip((target_long−target_short)/target_long,0,1)。每栋四类加和，街区内均值，再批内街区均值；不以建筑数量加权街区。匹配代价与训练损失一致；目标排列变化不影响集合损失。权重及beta锁定在config。

评价报告每街区：中心距离／L及米制；long/short相对误差；高度MAE米。重叠比例=(各矩形面积和−union面积)/面积和；越界比例=sum(rectangle−block面积)/面积和。预测及真实矩形目标各报一份。按街区均等权重汇总均值、中位数、Q1/Q3/IQR，按数量区间和城市给摘要，不作城市排名／显著性检验。逐案例A−B使用相同测试案例；仅位置误差不能宣告编码验证通过。

## 抽样、划分、冻结与暴露

冻结索引先筛8—64节点及明确过审的completed记录；按城市／8—16、17—32、33—64三档，以seed+sample_id稳定散列排序、轮换读取；每城最多读600图、选200例，总4800封顶，不保证凑满。几何／正高度／正面积不完整记子样本inapplicable，不改全量有效性。

同城1km格网（metric坐标bbox中心）、同城共享建筑ID、同城相同block_id、完全相同的尺度化矩形目标或米制矩形目标（均含轴向与log高度，6位小数、顺序无关）通过union-find合组；同时使用两种目标指纹，避免平移缩放复制案例遗漏。组ID为成员sample_id列表散列。seed+group_id散列映射约80/10/10；同组不可跨划分。分组内有旧Phase5、6A、6B或本轮短检查展示案例时，若分到test改为val；整体和逐城比例如实报告。任何暴露清单缺失／不可读都记录缺口，不声称完全未见。格网合并只是同城空间分组，不是严格空间独立或跨城泛化。空split拒绝训练，需要讨论，不自动变更划分规则。

先保存固定manifest和split，再仅用train拟合统计量，两模型共用同份缓存。run-id与配置、输入SHA、合同及关键代码散列绑定；改变配置／代码不能覆盖或继续原run。训练只读冻结的派生张量，评价再核对选中原Graph字节。仅latest/best checkpoint；显式--resume同一模型并恢复优化器/RNG/早停，已完成运行拒绝续训。GPU失败停止，不自动CPU长跑。两模型各一次首轮运行后讨论。

## 检查与状态

实施授权仅单元测试和≤8例、≤20优化步、≤300秒短检查；本实现使用4优化步总额。当前缺torch/PyG时仅执行数据检查，模型短检查待补，不能称“准备验收通过”。短检查不保存正式缓存、不作为效果证据。正式prepare、训练和测试图册由研究者手动启动。状态区分数据检查、短检查可运行、已训练、已评价、研究者接受；没有自动成功阈值。

依据：配套《第三章_现有图编码定稿与重构用途说明_v0.1.md》及阶段6C任务书。原件随本轮本地交接目录保留，位置及SHA见plan_review。
