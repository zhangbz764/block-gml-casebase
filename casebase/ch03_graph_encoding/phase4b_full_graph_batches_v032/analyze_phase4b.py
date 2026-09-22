"""Create compact summaries for one completed or accounted Phase 4B part."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PHASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PHASE_DIR / "results"


def _latest(status: pd.DataFrame) -> pd.DataFrame:
    frame = status.copy()
    frame["attempt"] = pd.to_numeric(frame["attempt"], errors="raise").astype(int)
    return frame.sort_values(["sample_id", "attempt"], kind="mergesort").drop_duplicates("sample_id", keep="last")


def summarize_status(status: pd.DataFrame, planned: int, skipped: int) -> dict:
    frame = _latest(status)
    for field in ("query_seconds", "encoding_seconds", "runtime_seconds", "output_bytes", "excluded_building_count"):
        frame[field] = pd.to_numeric(frame[field], errors="coerce").fillna(0)
    total_runtime = float(frame["runtime_seconds"].sum())
    query = float(frame["query_seconds"].sum())
    return {
        "planned": int(planned),
        "completed": int(frame["status"].eq("completed").sum()),
        "excluded_no_valid_building": int(frame["status"].eq("excluded_no_valid_building").sum()),
        "failed": int(frame["status"].eq("failed").sum()),
        "interrupted": int(planned - len(frame)),
        "skipped": int(skipped),
        "building_exclusions": int(frame["excluded_building_count"].sum()),
        "query_seconds": query,
        "encoding_seconds": float(frame["encoding_seconds"].sum()),
        "runtime_seconds": total_runtime,
        "output_bytes": int(frame["output_bytes"].sum()),
        "query_share": query / total_runtime if total_runtime > 0 else None,
    }


def analyze_part(part_id: str, planned: int, skipped: int = 0) -> dict:
    run_dir = RESULTS_DIR / "runs" / "full_v032" / part_id
    status = pd.read_csv(run_dir / "encoding_status.csv", encoding="utf-8-sig", low_memory=False)
    summary = {"part_id": part_id, **summarize_status(status, planned, skipped)}
    audit = RESULTS_DIR / "audit" / "full_v032" / part_id
    audit.mkdir(parents=True, exist_ok=True)
    (audit / "part_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    latest = _latest(status)
    latest.groupby(["city", "status"], sort=True).size().rename("case_count").reset_index().to_csv(
        audit / "status_by_city.csv", index=False, encoding="utf-8-sig"
    )
    latest.loc[latest["status"].eq("failed"), ["city", "block_id", "failure_type", "failure_message"]].to_csv(
        audit / "failure_summary.csv", index=False, encoding="utf-8-sig"
    )
    return summary
