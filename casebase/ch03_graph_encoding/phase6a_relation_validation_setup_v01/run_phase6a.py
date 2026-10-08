"""Manual Phase 6A entry point: frozen index -> selected case cards, no model run."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys

import pandas as pd

from model_inputs import BLOCK_FIELDS, NODE_CORE_FIELDS, NODE_DERIVED_FIELDS, EDGE_FIELDS, extract_model_inputs
from prepare_pairs import IDENTITY_FIELDS, COUNT_FIELDS, PAIR_COLUMNS, parse_bool, sample_development, select_candidate_pairs
from render_pairs import EXAMPLE_FIELDS, build_review_html, load_graph, metric_rows, render_pair


PHASE_DIR = Path(__file__).resolve().parent
CH03_DIR = PHASE_DIR.parent
PHASE5_DIR = CH03_DIR / "phase5_graph_dataset_audit_v01"
PHASE4B_DIR = CH03_DIR / "phase4b_full_graph_batches_v032"
REVIEW_FIELDS = ("pair_id", "metrics_close_enough", "layout_difference", "encoding_correspondence", "keep_for_thesis", "comment")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reserve_run_directory(results_root: Path, run_id: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", run_id):
        raise ValueError("run-id must contain only ASCII letters, digits, underscores and hyphens")
    path = Path(results_root) / "review" / run_id
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Run directory contains existing outputs; use a new run-id: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def validate_source_counts(index: pd.DataFrame, expected: dict) -> dict:
    counts = {str(key): int(value) for key, value in index["case_status"].value_counts().items()}
    actual = {"entries": len(index), "completed": counts.get("completed", 0),
              "excluded_no_valid_building": counts.get("excluded_no_valid_building", 0),
              "cities": int(index["city"].nunique())}
    if actual != expected or sum(counts.values()) != len(index) or set(counts) - {"completed", "excluded_no_valid_building"}:
        raise ValueError(f"Frozen index counts differ from the accepted dataset: {actual}; expected {expected}")
    identity_fields = ("sample_id", "city", "block_id", "part_id")
    if index[list(identity_fields)].isna().any().any() or index["sample_id"].duplicated().any():
        raise ValueError("Frozen index contains missing identities or duplicate sample IDs")
    # Excluded entries legitimately have neither a Graph nor propagated country metadata.
    if index.loc[index["case_status"].eq("completed"), ["country", "graph_path"]].isna().any().any():
        raise ValueError("Completed entries contain missing country metadata or Graph paths")
    return actual


def read_frozen_index(phase5_dir: Path, spec: dict) -> tuple[pd.DataFrame, dict]:
    path = phase5_dir / "results/dataset/graph_dataset_index.csv.gz"
    freeze_path = phase5_dir / "results/freeze.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8-sig"))
    expected_hash = freeze.get("output_sha256", {}).get("dataset/graph_dataset_index.csv.gz")
    actual_hash = sha256(path)
    if not expected_hash or actual_hash != expected_hash:
        raise ValueError("Phase 5 index SHA-256 does not match its freeze; upstream assets were not changed")
    fields = {*IDENTITY_FIELDS, *COUNT_FIELDS, *spec["pair_fields"], "case_status", "audit_passed"}
    # Read the compact index once; numeric fields are deliberately parsed by matching code.
    index = pd.read_csv(path, usecols=lambda name: name in fields,
                        dtype={name: "string" for name in IDENTITY_FIELDS}, low_memory=False)
    required = {*IDENTITY_FIELDS, *spec["pair_fields"], "case_status", "audit_passed"}
    if required - set(index):
        raise ValueError(f"Missing frozen index fields: {sorted(required - set(index))}")
    expected = {"entries": spec["expected_entries"], "completed": spec["expected_completed"],
                "excluded_no_valid_building": spec["expected_excluded"], "cities": spec["expected_cities"]}
    counts = validate_source_counts(index, expected)
    return index, {"index_path": str(path.resolve()), "index_sha256": actual_hash,
                   "freeze_path": str(freeze_path.resolve()), "freeze_sha256": sha256(freeze_path),
                   "accepted_counts": counts,
                   "historical_manual_review": "Accepted by 2026-10-04 review; do not reinterpret PENDING or mutate freeze"}


def write_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def write_table(path: Path, rows: list[dict], columns) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite: {path}")
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False, encoding="utf-8-sig")


def save_development(frame: pd.DataFrame, path: Path) -> None:
    content = frame.to_csv(index=False)
    if path.exists():
        if path.read_text(encoding="utf-8-sig") != content.replace("\r\n", "\n"):
            raise ValueError("Existing development_manifest differs; preserve the first selection and investigate")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8-sig", newline="") as handle:
        handle.write(content)


def run(mode: str, run_id: str, max_pairs: int | None = None) -> Path:
    spec = json.loads((PHASE_DIR / "pairing_spec_v01.json").read_text(encoding="utf-8"))
    cap = 2 if mode == "smoke" else 30
    if max_pairs is not None and not 1 <= max_pairs <= cap:
        raise ValueError(f"{mode} --max-pairs must be within 1..{cap}")
    spec["max_pairs"] = max_pairs if max_pairs is not None else cap
    if mode == "smoke":
        if not run_id.startswith("smoke_"):
            raise ValueError("Smoke run-id must start with smoke_ so preview outputs remain separately ignored")
        spec["max_per_city"] = spec["smoke_per_city"]
    elif run_id.startswith("smoke_"):
        raise ValueError("A formal candidates run must not use the smoke_ prefix")
    index, provenance = read_frozen_index(PHASE5_DIR, spec)
    development = sample_development(index, spec)
    sampling = dict(development.attrs["summary"])
    pairs = select_candidate_pairs(development, spec)
    sampling["pairing"] = pairs.attrs["summary"]
    root = PHASE_DIR / "results"
    if mode == "candidates":
        save_development(development, root / "sampling/development_manifest.csv")
    run_dir = reserve_run_directory(root, run_id)
    manifest = {"experiment_id": spec["experiment_id"], "mode": mode, "run_id": run_id,
                "generated_at_utc": datetime.now(timezone.utc).isoformat(), "encoding_version": spec["encoding_version"],
                **provenance, "graph_path_base": str(PHASE4B_DIR.resolve()),
                "pairing_spec": spec, "pairing_spec_sha256": sha256(PHASE_DIR / "pairing_spec_v01.json"),
                "generator_interface": "interface_unavailable", "ready_for_phase6b": False,
                "model_executed": False, "researcher_judgment": "not supplied"}
    write_json(run_dir / "input_manifest.json", manifest)
    write_json(run_dir / "sampling_summary.json", sampling)
    verification = {"program_checks_passed": False, "model_executed": False,
                    "researcher_judgment": "not supplied", "selected_pairs": len(pairs), "checked_graphs": 0}
    try:
        if mode == "smoke":
            save_development(development, run_dir / "development_manifest.csv")
        manifest_columns = [*PAIR_COLUMNS, *[f"{field}_{side}" for side in ("a", "b") for field in COUNT_FIELDS if f"{field}_{side}" in pairs]]
        write_table(run_dir / "pair_manifest.csv", pairs.to_dict("records"), manifest_columns)
        cards, relations, examples, metrics = [], [], [], []
        checks = []
        for pair in pairs.to_dict("records"):
            records = {}
            for side in ("a", "b"):
                lookup = {field: pair[f"{field}_{side}"] for field in ("sample_id", "city", "block_id", "graph_path")}
                record = load_graph(lookup, PHASE4B_DIR)
                inputs = extract_model_inputs(record)
                for field in ("block", "conditions"):
                    if not inputs["indicator"][field] == inputs["building_set"][field] == inputs["relation_graph"][field]:
                        raise ValueError(f"Comparison groups have inconsistent {field}")
                if inputs["building_set"]["nodes"] != inputs["relation_graph"]["nodes"]:
                    raise ValueError("Building-set and graph inputs have inconsistent common nodes")
                records[side] = record
                verification["checked_graphs"] += 1
            card = render_pair(records["a"], records["b"], pair, run_dir)
            cards.append(card); relations.extend(card["relation_rows"]); examples.extend(card["example_edge_rows"])
            metrics.extend(metric_rows(pair, spec)); checks.append({"pair_id": pair["pair_id"], "views": card["checks"]})
        build_review_html(pairs, cards, run_dir / "visual_review.html", spec, sampling)
        write_table(run_dir / "pair_metrics.csv", metrics,
                    ("pair_id", "city", "field", "label", "unit", "a_value", "b_value", "absolute_difference", "standardized_difference", "used_in_distance"))
        write_table(run_dir / "relation_summary.csv", relations,
                    ("pair_id", "side", "sample_id", "relation", "label", "unit", "edge_count", "valid_edges", "valid_values", "q1", "median", "q3", "iqr"))
        write_table(run_dir / "example_edges.csv", examples,
                    ("pair_id", "side", "sample_id", "label", "edge_id", "source", "target", *EXAMPLE_FIELDS, "missing_reasons"))
        write_table(run_dir / "pair_review.csv", [{"pair_id": pair_id} for pair_id in pairs["pair_id"]], REVIEW_FIELDS)
        verification.update({"program_checks_passed": True, "shared_inputs_equal": True,
                             "common_fields": {"block": len(BLOCK_FIELDS), "conditions": 3,
                                               "node_core": len(NODE_CORE_FIELDS), "node_derived": len(NODE_DERIVED_FIELDS), "edge": len(EDGE_FIELDS)},
                             "drawing_checks": checks, "is_morphology_or_model_proof": False})
    except Exception as error:
        verification["error"] = f"{type(error).__name__}: {error}"
        write_json(run_dir / "verification_summary.json", verification)
        raise
    write_json(run_dir / "verification_summary.json", verification)
    # Canonical formal metadata is published once; new runs retain their own full provenance.
    if mode == "candidates" and not (root / "input_manifest.json").exists():
        for name, value in (("input_manifest.json", manifest), ("sampling_summary.json", sampling), ("verification_summary.json", verification)):
            write_json(root / name, value)
    evidence = f"""# Phase 6A 本轮简报\n\n实验：{spec['experiment_id']}；模式：{mode}；run_id：{run_id}。\n\n冻结入口{len(index):,}例；开发池{len(development):,}例；入选{len(pairs)}对。分城/等级数量及不足原因见sampling_summary.json。候选在模型评分之前按固定规则生成；没有插补、扩大采样或迭代放宽阈值。\n\n页面：visual_review.html及images/；紧凑记录：pair_manifest.csv、pair_metrics.csv、relation_summary.csv、example_edges.csv。pair_review.csv未代填研究判断。索引SHA-256：`{provenance['index_sha256']}`。\n\n只读取{verification['checked_graphs']}个入选Graph；未改写上游、未训练模型、未输出全量张量。三组输入共享Block15及3项条件，集合/图共享Node8+3项派生值及缺失掩码，图另有Edge12。程序检查不构成形态判断或模型优越证据。\n\n当前生成器接口未提供，ready_for_phase6b=false。请选约8—12对或给一段总判断，并补充生成器路径/输入输出样例；本轮停止于案例及验证准备。\n"""
    with (run_dir / "phase6a_evidence_pack.md").open("x", encoding="utf-8") as handle:
        handle.write(evidence)
    if mode == "candidates" and not (root / "phase6a_evidence_pack.md").exists():
        with (root / "phase6a_evidence_pack.md").open("x", encoding="utf-8") as handle:
            handle.write(evidence)
    return run_dir / "visual_review.html"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("smoke", "candidates"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--max-pairs", type=int)
    args = parser.parse_args(argv)
    try:
        output = run(args.mode, args.run_id, args.max_pairs)
    except Exception as error:
        print(f"Phase 6A stopped: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(f"Open offline review: {output}")
    print("No model was trained; researcher judgment remains blank.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
