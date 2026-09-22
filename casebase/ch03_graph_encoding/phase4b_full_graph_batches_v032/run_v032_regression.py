"""Run the two frozen Phase 4B building-tolerance regression cases."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd


PHASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PHASE_DIR / "results"
CH03_DIR = PHASE_DIR.parent
PHASE4A_DIR = CH03_DIR / "phase4a_chunked_graph_pilot_v01"
PHASE3_DIR = CH03_DIR / "phase3_encoding_dataset_v03"
sys.path.insert(0, str(CH03_DIR)); sys.path.insert(0, str(PHASE4A_DIR)); sys.path.insert(0, str(PHASE3_DIR))

from batch_db_reader import fetch_block_batch, fetch_building_batch
from export_phase3 import DEFAULT_TESSELLATION_CONFIG, read_graph
from shared.db_readonly import readonly_connection
from export_phase4b_v032 import ENCODING_VERSION, encode_block_to_v032
from run_phase4b import atomic_write_graph


CASES = [
    {"sample_id": "P4B_US_BO_003310", "city": "Boston", "block_id": "US_BO_003310", "nodes": 37, "cells": 37, "edges": 71, "invalid": 1, "empty": 0},
    {"sample_id": "P4B_US_SF_000689", "city": "San Francisco", "block_id": "US_SF_000689", "nodes": 72, "cells": 72, "edges": 154, "invalid": 1, "empty": 4},
]


def main() -> int:
    summary_path = RESULTS_DIR / "audit" / "prep" / "v032_regression_summary.csv"
    graph_dir = RESULTS_DIR / "regression" / "graphs"
    occupied = [path for path in [summary_path, *[graph_dir / f"{case['sample_id']}.graph.json.gz" for case in CASES]] if path.exists()]
    if occupied:
        raise FileExistsError(f"Refusing to overwrite regression outputs: {occupied}")
    rows = []
    with readonly_connection(statement_timeout_ms=600_000) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SHOW transaction_read_only")
            read_only = cursor.fetchone()[0]
        if str(read_only).lower() != "on":
            raise RuntimeError("Regression database transaction is not read-only")
        for case in CASES:
            started = time.perf_counter()
            blocks = fetch_block_batch(connection, case["city"], [case["block_id"]])
            block = blocks[case["block_id"]]
            buildings = fetch_building_batch(connection, block["city_query"], int(block["metric_srid"]), [case["block_id"]])[case["block_id"]]
            outcome = encode_block_to_v032(case, block, buildings, sample_manifest_sha256="regression-v032", tessellation_config=DEFAULT_TESSELLATION_CONFIG)
            if outcome.status != "completed" or outcome.record is None:
                raise RuntimeError(f"Regression case did not complete: {case['block_id']}")
            counts = outcome.record["graph"]["excluded_building_reason_counts"]
            actual = {
                "nodes": len(outcome.record["nodes"]), "cells": len(outcome.record["cells"]), "edges": len(outcome.record["edges"]),
                "invalid": int(counts.get("invalid_polygonal_area_before_shrink", 0)), "empty": int(counts.get("empty_after_shrink", 0)),
            }
            passed = all(actual[key] == case[key] for key in actual) and outcome.record["validation"]["passed"]
            if not passed:
                raise RuntimeError(f"Regression counts changed for {case['block_id']}: {actual}")
            outcome.record["phase4b"] = {"regression": True, "encoding_version": ENCODING_VERSION, "block_id": case["block_id"]}
            path = graph_dir / f"{case['sample_id']}.graph.json.gz"
            atomic_write_graph(outcome.record, path)
            if not read_graph(path)["validation"]["passed"]:
                raise RuntimeError("Regression graph failed roundtrip validation")
            rows.append({"city": case["city"], "block_id": case["block_id"], "status": "completed", "node_count": actual["nodes"], "cell_count": actual["cells"], "edge_count": actual["edges"], "invalid_polygonal_area_before_shrink": actual["invalid"], "empty_after_shrink": actual["empty"], "validation_passed": True, "transaction_read_only": read_only, "runtime_seconds": round(time.perf_counter() - started, 6)})
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(summary_path, index=False, encoding="utf-8-sig", lineterminator="\n")
    print(json.dumps({"cases": len(rows), "passed": True, "transaction_read_only": read_only}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
