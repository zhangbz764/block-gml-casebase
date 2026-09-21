"""Local v0.3.1 graph revision that excludes collapsed building footprints."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal


PHASE_DIR = Path(__file__).resolve().parent
PHASE3_DIR = PHASE_DIR.parent / "phase3_encoding_dataset_v03"
sys.path.insert(0, str(PHASE3_DIR))

from export_phase3 import (
    DEFAULT_TESSELLATION_CONFIG,
    TessellationConfig,
    encode_block_to_v03,
    validate_v03_record,
)
from filter_buildings_v031 import partition_buildings_for_encoding


ENCODING_VERSION = "0.3.1"
EXPERIMENT_ID = "CH3-P3.1-NODE-CLEANUP-V031"


@dataclass(frozen=True)
class EncodingOutcome:
    status: Literal["completed", "excluded_no_valid_building"]
    record: dict | None
    excluded_buildings: list[dict]


def _enrich_exclusions(
    exclusions: list[dict[str, Any]],
    sample: dict[str, Any],
    block_record: dict[str, Any],
) -> list[dict[str, Any]]:
    return [
        {
            "sample_id": str(sample["sample_id"]),
            "city": str(block_record["city"]),
            "block_id": str(block_record["block_id"]),
            **item,
        }
        for item in exclusions
    ]


def _exclusion_summary(
    source_building_count: int,
    kept: list[dict[str, Any]],
    exclusions: list[dict[str, Any]],
) -> dict[str, Any]:
    excluded_area = sum(float(item["footprint_area_m2"]) for item in exclusions)
    encoded_area = sum(float(item["geometry"].area) for item in kept)
    source_area = encoded_area + excluded_area
    return {
        "source_building_count": source_building_count,
        "encoded_building_count": len(kept),
        "excluded_building_count": len(exclusions),
        "excluded_footprint_area_m2": excluded_area,
        "excluded_footprint_area_ratio": excluded_area / source_area if source_area > 0 else None,
    }


def encode_block_to_v031(
    sample: dict,
    block_record: dict,
    building_records: list[dict[str, Any]],
    *,
    sample_manifest_sha256: str,
    tessellation_config: TessellationConfig = DEFAULT_TESSELLATION_CONFIG,
) -> EncodingOutcome:
    """Exclude collapsed footprints before reusing the v0.3 encoding flow."""
    kept, exclusions = partition_buildings_for_encoding(
        building_records,
        block_record["geometry"],
        tessellation_config.shrink_m,
    )
    enriched = _enrich_exclusions(exclusions, sample, block_record)
    summary = _exclusion_summary(len(building_records), kept, enriched)
    if not kept:
        return EncodingOutcome(
            status="excluded_no_valid_building",
            record=None,
            excluded_buildings=enriched,
        )

    record = encode_block_to_v03(
        sample,
        block_record,
        kept,
        sample_manifest_sha256=sample_manifest_sha256,
        tessellation_config=tessellation_config,
    )
    repeated = [
        item
        for item in record.get("cell_failures", [])
        if item.get("failure_type") == "empty_after_shrink"
    ]
    if repeated:
        ids = ", ".join(str(item["building_id"]) for item in repeated)
        raise RuntimeError(f"Pre-filter/tessellation mismatch for building IDs: {ids}")

    record["encoding_version"] = ENCODING_VERSION
    record["experiment_id"] = EXPERIMENT_ID
    record["graph"].update(summary)
    record["exclusions"] = {"buildings": enriched, "summary": summary}

    excluded_ids = {str(item["building_id"]) for item in enriched}
    encoded_ids = {str(item["building_id"]) for item in record["nodes"]}
    cell_ids = {str(item["building_id"]) for item in record["cells"]}
    edge_ids = {
        str(endpoint)
        for edge in record["edges"]
        for endpoint in (edge["source"], edge["target"])
    }
    if excluded_ids & (encoded_ids | cell_ids | edge_ids):
        raise RuntimeError("Excluded building ID leaked into node, cell, or edge output")

    record["validation"] = validate_v03_record(record)
    return EncodingOutcome(
        status="completed",
        record=record,
        excluded_buildings=enriched,
    )
