# 编码修订v0.3.1

实验ID：`CH3-P3.1-NODE-CLEANUP-V031`

## 修订规则

1. 在调用阶段2的`assemble_graph`之前，对原始建筑底面执行与阶段2.5 tessellation一致的面状几何处理。
2. 面状几何按固定参数`0.4 m`向内缓冲；缓冲后为空或面积非正的对象标记为`empty_after_shrink`并排除。
3. 不增加面积、周长、紧凑度等阈值，不执行`0.4 → 0 m`回退。
4. 排除对象不生成node、cell或edge；若街区没有剩余建筑，不写空graph，状态为`excluded_no_valid_building`。
5. 若仅剩一个建筑，保留为1 node、1 cell、0 edge的合法图。
6. 成功图使用顶层`exclusions`记录逐对象信息和汇总，不将排除对象写入`nodes`或`cell_failures`。

## 审计量定义

- `source_building_count`：数据库查询返回的原始建筑记录数。
- `encoded_building_count`：预排除后进入node派生的建筑数。
- `excluded_building_count`：因`empty_after_shrink`被排除的建筑数。
- `excluded_footprint_area_m2`：被排除对象在几何修复后的原始底面面积之和。
- `excluded_footprint_area_ratio`：上述面积除以全部可判定源建筑底面面积之和；分母为0时记为null。

v0.3.1对应57个受影响街区的修订结果，与旧版本图分别保存。
