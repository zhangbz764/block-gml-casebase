"""Manual-only entry point for seven probes and up to three existing case cards."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys

from control_probes import FIELDS, GROUPS, PHASE4B, PHASE6A, encode_probe, file_sha256, make_probe_layouts, select_real_cases, summarize_probe
from render_control_probes import OPERABLE, describe_response, render_preview

PHASE = Path(__file__).resolve().parent
RESULTS = PHASE / "results"


def reserve_output_dirs(root: Path, run_id: str) -> tuple[Path, Path]:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", run_id):
        raise ValueError("Unsafe run-id: use ASCII letters, digits, underscores and hyphens")
    probes, review = root / "probes" / run_id, root / "review" / run_id
    for path in (probes, review):
        if path.exists() and any(path.iterdir()):
            raise FileExistsError(f"Existing outputs are preserved; use a new run-id: {path}")
    for path in (probes, review):
        path.mkdir(parents=True, exist_ok=True)
    return probes, review


def write_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def write_text(path: Path, value: str) -> None:
    with path.open("x", encoding="utf-8") as handle:
        handle.write(value)


def decision_text(records: dict, failures: list, interface_status: str) -> str:
    sections = []
    for target, title in (("gap", "建筑间隔"), ("offset", "两排行列错位"), ("height", "高层体量布置")):
        subset = {name: records[name] for name in GROUPS[target] if name in records}
        sections.append(f"## {title}\n\n观察（仅已完成布局）：{describe_response(subset, target)}\n\n"
                        f"已有记录支持端点层面的表达；设计者可提出：{OPERABLE[target]}不能由分布摘要自动得到完整设计条件。\n")
    height_pairs = {}
    for name in GROUPS["height"]:
        if name not in records:
            continue
        record = records[name]
        high_ids = {node["building_id"] for node in record["nodes"] if node.get("height_m") == 24}
        height_pairs[name] = [(edge["source"],edge["target"]) for edge in record["edges"]
                              if edge["source"] in high_ids and edge["target"] in high_ids]
    failures_text = "无编码失败。" if not failures else "失败原样保留："+"；".join(f"{row['layout_id']}: {row['error']}" for row in failures)
    return "# 6B前置：控制准备短决策说明\n\n状态：小试已执行，分析仅描述本轮输出；不是生成或模型有效性结论。研究者判断待提供。\n\n"+"\n".join(sections)+f"""
两种高度配置的24m—24m具体邻接：{json.dumps(height_pairs, ensure_ascii=False)}。两例高度集合相同，平面不变；整体统计不说明高层位于哪里。

## 推荐与暂缓

建议优先讨论“建筑间隔”，属于暂定操作目标，不是依据显著性筛选出的优胜方案。理由是固定轮廓与边界下，位置调整、指定端点间距与约束核对较明确，可先用几何/规则程序处理；若本轮缺失或响应不足，须先讨论适用性，不能据此启动正式实验。两排行列错位暂缓，因为长边投影重叠和连线夹角不等于完整行列类型，long_side_offset_norm主要反映主轴法向间隔，并非本次横向错位；高度布置暂缓，因为还需明确参考案例到目标建筑ID的对应及高体量分配操作。

## 编码怎样参与下一步

从真实参考Graph提取设计者选定的建筑对、轮廓、间隔记录及场地/尺度条件，将这些端点关系转成位置调整的目标约束，再由可调用的几何或生成接口实施调整，最后重编码核对变化、边界冲突和未满足关系。编码既组织参考条件和操作约束，也检查响应，不只是事后评价指标。这次仅直接构造七个布局并编码，没有实现参考Graph到新场地的生成或完整反演。

目前无需强行训练：给定位置的间距与对齐属性可由几何直算，图学习是否有必要仍待检验。若以后采用模型，预测/生成对象应为可执行的位置调整或布局，标签应由独立的受约束布局/操作样本取得；输入可含参考图和目标条件，但不得直接提供目标输出位置后预测自身边属性来证明用途。数据划分、控制条件公平性和独立评价需在正式任务前确定，本轮不训练、不扩充数据集。

## 能力缺口与停止线

生成器盘点状态：{interface_status}。可确定编码目标，生成实施待补接口；ready_for_formal_phase6b=false。最少需要：明确可复用的代码路径或输入输出样例，输入场地边界、建筑ID/轮廓/高度、可调整位置及固定条件、端点对应的目标间隔，输出调整后轮廓/位置/高度及不可行原因。优先核实能否按这些条件操纵布局，不自行重建simforms/flexurban。

{failures_text} 本地米制坐标使用未指定SRID 0，不是EPSG或真实城市。真实卡只读当前6A清单中的已看案例，不是监督盲测，无生成后图。局部可读性不证明GNN更优、图不可替代、真实转译成功或设计者主体性提高。交付后停止，研究者通过对话决定目标及解释边界。
"""


def prepare(run_id: str) -> tuple[Path, bool]:
    spec_path = PHASE / "probe_layouts_v01.json"
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    layouts = make_probe_layouts(spec)
    manifest_path = PHASE6A / "results/review/candidate_v01/pair_manifest.csv"
    inventory_path = PHASE6A / "results/generator_interface_inventory.csv"
    # Read just the previous inventory, never search disks or external generator platforms.
    with inventory_path.open(encoding="utf-8-sig", newline="") as handle:
        inventory = list(csv.DictReader(handle))
    interface_status = ";".join(sorted({row.get("status", "unknown") for row in inventory})) or "unknown"
    probes, review = reserve_output_dirs(RESULTS, run_id)
    inputs = [spec_path, manifest_path, inventory_path, PHASE6A / "model_input_fields_v01.csv",
              PHASE4B / "export_phase4b_v032.py", PHASE.parent / "phase5_graph_dataset_audit_v01/feature_registry_snapshot_v03.tsv"]
    metadata = {"task_id": spec["task_id"], "run_id": run_id, "started_at_utc": datetime.now(timezone.utc).isoformat(),
                "state": "RUNNING", "encoding_version": "0.3.2", "source_kind": "synthetic_local",
                "graph_path_base": str(PHASE4B.resolve()), "interface_status": interface_status,
                "layout_definition_sha256": layouts[0]["definition_sha256"], "model_or_generator_run": False,
                "ready_for_formal_phase6b": False, "source_graphs_copied": False, "researcher_judgment": "not supplied"}
    records, responses, failures = {}, [], []
    try:
        metadata["read_inputs"] = [{"path": str(path.resolve()), "sha256": file_sha256(path)} for path in inputs]
        graph_dir = probes / "graphs"; graph_dir.mkdir()
        for layout in layouts:
            try:
                record = encode_probe(layout)
            except Exception as error:
                row = {"layout_id": layout["id"], "status": "failed", "error": f"{type(error).__name__}: {error}"}
                responses.append(row); failures.append(row)
                continue  # Exactly these seven layouts, no repaired variants or parameter search.
            records[layout["id"]] = record
            responses.append(summarize_probe(record))
            write_json(graph_dir / f"{layout['id']}.json", record)
        columns = ["layout_id", "status", "node_count", "edge_count", "area_m2", "footprint_area_total_m2",
                   "avh_w_m", "bcr", "cell_failure_count", *[f"{field}_{stat}" for field in FIELDS for stat in ("valid_count", "median", "iqr")], "error"]
        with (probes / "probe_response.csv").open("x", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns); writer.writeheader(); writer.writerows(responses)
        real_cards, real_sources = select_real_cases(manifest_path, PHASE4B)
        metadata.update({"read_real_sources": real_sources,
                         "selected_real_ids": {target: card["source"]["sample_id"] if card["source"] else None for target,card in real_cards.items()},
                         "real_selection_notes": {target: card["reason"] for target,card in real_cards.items()}})
        metadata["drawing_checks"] = render_preview(records, real_cards, review)
        metadata.update({"state": "EXECUTED_WITH_FAILURES" if failures else "EXECUTED",
                         "encoded_probe_count": len(records), "failures": failures,
                         "finished_at_utc": datetime.now(timezone.utc).isoformat()})
        decision = decision_text(records, failures, interface_status)
        write_text(probes / "control_preparation_decision_summary.md", decision)
        evidence = f"""# 6B前置运行简报\n\n命令：run_phase6b_preparation.py --mode prepare --run-id {run_id}。状态：{metadata['state']}。\n\n指定七个布局，完成{len(records)}个小Graph；失败{len(failures)}个，原因见probe_response.csv。未改源Graph或数据库，未训练或调用生成器。真实记录只从当前6A清单读取{len(real_sources)}个，展示ID见input_manifest.json，不扩展案例池。\n\n页面：results/review/{run_id}/visual_review.html及images/，必须一起交付。响应：results/probes/{run_id}/probe_response.csv；具体端点保存在同目录graphs/*.json。运行输入、源文件散列和绘图范围见同目录input_manifest.json。\n\n建议与缺口：control_preparation_decision_summary.md；推荐暂先讨论建筑间隔，真实生成实施待补接口。程序只完成输出，不代填形态判断；未进行执行者目视检查，应打开页面检查字体、比例、图例和轴测。\n\nGit忽略HTML、images/、graphs/及内部决策/证据说明；Graph和图片可从配置重建，审核数据包须包含它们。未建立额外备份、训练数据或临时脚本。\n"""
        write_text(probes / "phase6b_preparation_evidence_pack.md", evidence)
        write_json(probes / "input_manifest.json", metadata)
        # The first completed preview owns canonical root records; later previews remain per-run.
        if not (RESULTS / "input_manifest.json").exists():
            write_json(RESULTS / "input_manifest.json", metadata)
            write_text(RESULTS / "control_preparation_decision_summary.md", decision)
            write_text(RESULTS / "phase6b_preparation_evidence_pack.md", evidence)
        return review / "visual_review.html", not failures
    except Exception as error:
        metadata.update({"state": "FAILED", "error": f"{type(error).__name__}: {error}",
                         "encoded_probe_count": len(records), "failures": failures})
        if not (probes / "input_manifest.json").exists():
            write_json(probes / "input_manifest.json", metadata)
        raise


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("prepare",))
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)
    try:
        page, passed = prepare(args.run_id)
    except Exception as error:
        print(f"Preparation stopped: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(f"Preview: {page}")
    print("No model/generator run; open the page before judging morphology.")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
