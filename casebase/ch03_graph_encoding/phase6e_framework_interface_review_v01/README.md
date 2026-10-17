# Phase 6E 框架选型与编码接口审查

范围：有限审查GlobalMapper、HouseDiffusion的官方源码与现有v0.3.2接口，给出一条下一步路线；不是实现生成器或验证生成效果。

交付：

- `results/framework_interface_review.md`：链路、接口、推荐及实施前缺口，入Git。
- `results/source_inventory.csv`：源码URL、固定提交、访问日期、可见资源及缺口，入Git。
- `results/bulk/upstream/`：24份固定版本官方文本缓存，224,416字节；已有bulk规则忽略，可重新取得。不是权重、数据集或临时备份；清理时可按此目录整体讨论。

2026-10-15只读审查。最多回读6D既有3例输入，实际原Graph只读Boston一例；索引仅读表头。不连接数据库，不扫描全库几何，不重审6C／6D结果，不运行上游入口或网络前向，不安装、不下载权重／数据、不commit／push，不改全项目.gitignore。

任务书指定executing-plans且不使用子代理；本次按只读报告任务执行，不新增测试、实现合同、freeze或内部审查台账。判断和未确认边界合并在主报告。通用实施流程中的代码测试、分支收尾与提交不适用本次授权。

建议为有前置门槛的GlobalMapper结构适配路线：保留槽位生成骨架，独立接入案例关系记忆和新设计条件；不是现成可跑的承诺。GlobalMapper源码授权及本地几何／槽位适配须在实施前解决，研究者确认前不实施。
