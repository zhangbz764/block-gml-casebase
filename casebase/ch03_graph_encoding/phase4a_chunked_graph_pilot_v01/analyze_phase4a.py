"""Analyze Phase 4A status records and produce stratified resource projections."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


PHASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PHASE_DIR / "results"


def project_resources(
    status: pd.DataFrame,
    weight_multiplier: float = 1.0,
) -> dict[str, float]:
    """Project totals with the frozen city × stratum sampling weights."""
    frame = status.copy()
    numeric = [
        "sample_weight", "query_seconds", "encoding_seconds", "runtime_seconds",
        "output_bytes", "python_peak_memory_mb",
    ]
    for field in numeric:
        frame[field] = pd.to_numeric(frame[field], errors="coerce").fillna(0.0)
    frame["projection_weight"] = frame["sample_weight"] * float(weight_multiplier)
    completed = frame["status"].eq("completed")
    return {
        "projected_query_seconds": float((frame["query_seconds"] * frame["projection_weight"]).sum()),
        "projected_encoding_seconds": float((frame["encoding_seconds"] * frame["projection_weight"]).sum()),
        "projected_total_runtime_seconds": float((frame["runtime_seconds"] * frame["projection_weight"]).sum()),
        "projected_graph_bytes": float((frame.loc[completed, "output_bytes"] * frame.loc[completed, "projection_weight"]).sum()),
        "projected_graph_file_count": float(frame.loc[completed, "projection_weight"].sum()),
        "peak_python_memory_mb": float(frame["python_peak_memory_mb"].max()),
    }


def _latest_status(status: pd.DataFrame) -> pd.DataFrame:
    frame = status.copy()
    frame["attempt"] = pd.to_numeric(frame["attempt"], errors="raise").astype(int)
    return (
        frame.sort_values(["sample_id", "attempt"], kind="mergesort")
        .drop_duplicates("sample_id", keep="last")
        .reset_index(drop=True)
    )


def analyze(mode: str, run_id: str) -> dict[str, Any]:
    run_root = RESULTS_DIR / mode / run_id
    audit_dir = RESULTS_DIR / "audit" / run_id
    if audit_dir.exists() and any(audit_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite audit: {audit_dir}")
    status = _latest_status(pd.read_csv(run_root / "encoding_status.csv", encoding="utf-8-sig"))
    chunks = pd.read_csv(run_root / "chunk_status.csv", encoding="utf-8-sig")
    weight_multiplier = 25.0 if mode == "smoke" else 1.0
    projection = project_resources(status, weight_multiplier=weight_multiplier)
    projection["weight_multiplier"] = weight_multiplier
    projection["query_share_of_projected_runtime"] = (
        projection["projected_query_seconds"] / projection["projected_total_runtime_seconds"]
        if projection["projected_total_runtime_seconds"] > 0 else None
    )

    grouped_rows = []
    for (city, stratum), group in status.groupby(["city", "city_count_stratum"], sort=True):
        subtotal = project_resources(group, weight_multiplier=weight_multiplier)
        grouped_rows.append(
            {
                "city": city,
                "city_count_stratum": stratum,
                "sample_count": len(group),
                "completed": int(group["status"].eq("completed").sum()),
                "excluded_no_valid_building": int(group["status"].eq("excluded_no_valid_building").sum()),
                "failed": int(group["status"].eq("failed").sum()),
                "excluded_building_count": int(pd.to_numeric(group["excluded_building_count"], errors="coerce").fillna(0).sum()),
                **subtotal,
            }
        )
    runtime = pd.DataFrame(grouped_rows)
    failures = (
        status.loc[status["status"].eq("failed")]
        .groupby(["city", "city_count_stratum", "failure_type", "failure_message"], dropna=False, sort=True)
        .size()
        .rename("case_count")
        .reset_index()
    )
    timeout_count = int(
        status["failure_message"].astype("string").str.contains("timeout", case=False, na=False).sum()
    )
    query_share = projection["query_share_of_projected_runtime"] or 0.0
    materialize = timeout_count > 0 or query_share >= 0.5
    assessment = {
        "run_id": run_id,
        "timeout_failure_count": timeout_count,
        "query_share_of_projected_runtime": query_share,
        "recommendation": (
            "consider_materialized_helper_table_next_phase"
            if materialize else "continue_csv_driven_readonly_batches"
        ),
        "table_created_in_phase4a": False,
    }
    expected_rows = 96 if mode == "smoke" else 2400
    covered_cells = len(status[["city", "city_count_stratum"]].drop_duplicates())
    issues = []
    if len(status) != expected_rows:
        issues.append(f"expected {expected_rows} status rows, got {len(status)}")
    if covered_cells != 96:
        issues.append(f"expected 96 city-stratum cells, got {covered_cells}")
    if status["failure_message"].isna().where(status["status"].eq("failed"), False).any():
        issues.append("a failed case has no failure message")
    verification = {
        "run_id": run_id,
        "mode": mode,
        "passed": not issues,
        "planned_rows": expected_rows,
        "status_rows": len(status),
        "city_count": int(status["city"].nunique()),
        "city_stratum_cells": covered_cells,
        "completed": int(status["status"].eq("completed").sum()),
        "excluded_no_valid_building": int(status["status"].eq("excluded_no_valid_building").sum()),
        "failed": int(status["status"].eq("failed").sum()),
        "issues": issues,
    }
    audit_dir.mkdir(parents=True, exist_ok=True)
    runtime.to_csv(audit_dir / "runtime_by_city_stratum.csv", index=False, encoding="utf-8-sig")
    chunks.to_csv(audit_dir / "query_performance.csv", index=False, encoding="utf-8-sig")
    failures.to_csv(audit_dir / "failure_summary.csv", index=False, encoding="utf-8-sig")
    for name, value in (
        ("resource_projection.json", projection),
        ("db_access_assessment.json", assessment),
        ("verification_summary.json", verification),
    ):
        (audit_dir / name).write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps({**verification, **projection, "db_recommendation": assessment["recommendation"]}, ensure_ascii=False))
    return verification


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "pilot"), required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    return 0 if analyze(args.mode, args.run_id)["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
