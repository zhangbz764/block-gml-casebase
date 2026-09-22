"""v0.3.2 graph export with narrowly defined building-level tolerance."""

from __future__ import annotations

import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal


PHASE_DIR = Path(__file__).resolve().parent
PHASE3_DIR = PHASE_DIR.parent / "phase3_encoding_dataset_v03"
sys.path.insert(0, str(PHASE3_DIR))

from export_phase3 import DEFAULT_TESSELLATION_CONFIG, TessellationConfig, encode_block_to_v03, validate_v03_record
from filter_buildings_v032 import partition_buildings_for_encoding


ENCODING_VERSION = "0.3.2"
EXPERIMENT_ID = "CH3-P4B-FULL-GRAPH-V032"


@dataclass(frozen=True)
class EncodingOutcome:
    status: Literal["completed", "excluded_no_valid_building"]
    record: dict | None
    excluded_buildings: list[dict]


def _enrich(exclusions: list[dict], sample: dict, block: dict) -> list[dict]:
    return [
        {
            "sample_id": str(sample["sample_id"]),
            "city": str(block["city"]),
            "block_id": str(block["block_id"]),
            **item,
        }
        for item in exclusions
    ]


def encode_block_to_v032(
    sample: dict,
    block_record: dict,
    building_records: list[dict[str, Any]],
    *,
    sample_manifest_sha256: str,
    tessellation_config: TessellationConfig = DEFAULT_TESSELLATION_CONFIG,
) -> EncodingOutcome:
    kept, raw_exclusions = partition_buildings_for_encoding(
        building_records, block_record["geometry"], tessellation_config.shrink_m
    )
    exclusions = _enrich(raw_exclusions, sample, block_record)
    if not kept:
        return EncodingOutcome("excluded_no_valid_building", None, exclusions)

    record = encode_block_to_v03(
        sample,
        block_record,
        kept,
        sample_manifest_sha256=sample_manifest_sha256,
        tessellation_config=tessellation_config,
    )
    reason_counts = Counter(item["reason"] for item in exclusions)
    summary = {
        "source_building_count": len(building_records),
        "encoded_building_count": len(kept),
        "excluded_building_count": len(exclusions),
        "excluded_building_reason_counts": dict(sorted(reason_counts.items())),
    }
    if summary["source_building_count"] != summary["encoded_building_count"] + summary["excluded_building_count"]:
        raise RuntimeError("Building count conservation failed")
    record["encoding_version"] = ENCODING_VERSION
    record["experiment_id"] = EXPERIMENT_ID
    record["graph"].update(summary)
    record["exclusions"] = {"buildings": exclusions, "summary": summary}

    excluded_ids = {str(item["building_id"]) for item in exclusions}
    encoded_ids = {str(item["building_id"]) for item in record["nodes"]}
    cell_ids = {str(item["building_id"]) for item in record["cells"]}
    edge_ids = {str(endpoint) for edge in record["edges"] for endpoint in (edge["source"], edge["target"])}
    if excluded_ids & (encoded_ids | cell_ids | edge_ids):
        raise RuntimeError("Excluded building leaked into graph output")
    record["validation"] = validate_v03_record(record)
    return EncodingOutcome("completed", record, exclusions)
