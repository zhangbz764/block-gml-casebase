"""Generate the short Chinese interpretation sheet for a completed Phase 5 audit."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


TAIL_FIELDS = {
    "node_count_p99": "节点数P99",
    "edge_count_p99": "边数P99",
    "connected_components_p99": "连通分量P99",
    "isolated_node_ratio_p99": "孤立节点比例P99",
    "combined_weak_candidate_ratio_p99": "组合弱边比例P99",
    "excluded_building_ratio_p99": "建筑排除比例P99",
}


def _integer(value) -> str:
    return f"{int(value or 0):,}"


def build_researcher_summary(
    global_summary: dict,
    city_summary: pd.DataFrame,
    verification: dict,
    *,
    expected_entries: int,
) -> str:
    entries = int(global_summary.get("entry_count", 0))
    completed = int(global_summary.get("completed_graph_count", 0))
    excluded = int(global_summary.get("excluded_no_valid_building_count", 0))
    matched = entries == int(expected_entries) and completed + excluded == int(expected_entries)
    hard_issues = int(global_summary.get("hard_issue_entry_count", 0))
    automated = verification.get("automated_audit_passed") is True
    consistency = "自动一致性检查通过" if automated and matched and hard_issues == 0 else "自动一致性检查未通过，需先查看verification_summary.json"

    tail_lines: list[str] = []
    for field, label in TAIL_FIELDS.items():
        if field not in city_summary or city_summary[field].dropna().empty:
            continue
        values = pd.to_numeric(city_summary[field], errors="coerce")
        index = values.idxmax()
        tail_lines.append(f"- {label}最高：{city_summary.loc[index, 'city']}（{values.loc[index]:.4g}）。")
    if not tail_lines:
        tail_lines.append("- 暂无可用的城市尾部统计。")

    lines = [
        "# Phase 5全量图数据集审计：研究者摘要",
        "",
        "## 一、全量入口是否闭合",
        "",
        f"- 冻结范围应有{_integer(expected_entries)}个入口，本次索引得到{_integer(entries)}个；"
        f"其中{_integer(completed)}个生成Graph，{_integer(excluded)}个因无有效建筑而不生成Graph。",
        f"- 入口记账{'完整' if matched else '不完整'}；{consistency}。",
        f"- 发现读取或结构硬错误的入口数：{_integer(hard_issues)}。",
        "",
        "## 二、对象总体规模",
        "",
        f"- Node总数：{_integer(global_summary.get('node_total'))}。",
        f"- Edge总数：{_integer(global_summary.get('edge_total'))}。",
        f"- Cell failure总数：{_integer(global_summary.get('cell_failure_total'))}。这表示个别Node没有有效Cell，但已有明确失败记录，并非无声遗漏。",
        f"- 编码前排除的建筑总数：{_integer(global_summary.get('excluded_building_total'))}。",
        "",
        "## 三、城市结构尾部提示",
        "",
        "以下城市只是各指标P99尾部最明显的检查入口，不等同于错误或应删除样本：",
        "",
        *tail_lines,
        "",
        "## 四、需要研究者判断",
        "",
        "1. 代表案例中的连接关系是否符合建筑形态直觉；",
        "2. 极端案例是合理形态，还是源数据或街区切分问题；",
        "3. 是否存在某一城市的系统性异常；",
        "4. 这些结果是否足以把v0.3.2作为后续实验的冻结输入。",
        "",
        "不需要逐项理解散列、恢复签名等内部字段，也不要求逐行填写48例。若总体正常，只需在阶段判断中写一段总意见；发现问题时再记录对应案例。",
    ]
    return "\n".join(lines) + "\n"


def write_researcher_summary(
    global_summary: dict,
    city_summary: pd.DataFrame,
    verification: dict,
    output_path: Path,
    *,
    expected_entries: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp")
    temporary.write_text(build_researcher_summary(global_summary, city_summary, verification,
                                                   expected_entries=expected_entries), encoding="utf-8")
    os.replace(temporary, output_path)
