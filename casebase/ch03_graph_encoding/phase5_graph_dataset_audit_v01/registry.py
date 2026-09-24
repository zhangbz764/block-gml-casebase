"""Load and validate the frozen Phase 5 audit specification and field registry."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


REGISTRY_SHA256 = "88b0ed3b019161c32077672057f8d696b06ef8d1ee480159a0744f874cc7253a"
REGISTRY_COLUMNS = [
    "entity", "field", "role", "unit", "transform_or_encoding",
    "missing_rule", "default_input", "rationale",
]
EXPECTED_ENTITY_COUNTS = {"graph": 21, "block": 26, "node": 26, "cell": 5, "edge": 25}
EXPECTED_ROLE_COUNTS = {
    "auxiliary_interpretation": 33,
    "core_model": 35,
    "provenance_trace": 21,
    "quality_audit": 14,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_audit_spec(path: Path) -> dict:
    spec = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "experiment_id", "encoding_version", "expected_entries", "expected_cities",
        "part_ids", "phase35b_manifest_sha256", "phase4b_code_bundle_sha256",
        "registry_sha256", "formal_run_id",
    }
    missing = sorted(required - set(spec))
    if missing:
        raise ValueError(f"Audit specification is missing keys: {missing}")
    if spec["part_ids"] != [f"part_{index:02d}" for index in range(1, 13)]:
        raise ValueError("Audit specification part IDs are not the frozen twelve-part sequence")
    return spec


def load_registry(path: Path, expected_sha256: str | None = REGISTRY_SHA256) -> pd.DataFrame:
    if expected_sha256 is not None and sha256_file(path) != expected_sha256:
        raise ValueError("Feature registry SHA256 differs from the frozen v0.3 registry")
    frame = pd.read_csv(path, sep="\t", encoding="utf-8")
    if list(frame.columns) != REGISTRY_COLUMNS:
        raise ValueError(f"Feature registry columns changed: {list(frame.columns)}")
    if frame["field"].duplicated().any():
        duplicates = sorted(frame.loc[frame["field"].duplicated(keep=False), "field"].unique())
        raise ValueError(f"Feature registry contains duplicate logical fields: {duplicates}")
    if len(frame) != 103:
        raise ValueError(f"Feature registry must contain 103 rows, got {len(frame)}")
    summary = registry_summary(frame)
    if summary["entity_counts"] != EXPECTED_ENTITY_COUNTS:
        raise ValueError(f"Feature entity counts changed: {summary['entity_counts']}")
    if summary["role_counts"] != EXPECTED_ROLE_COUNTS:
        raise ValueError(f"Feature role counts changed: {summary['role_counts']}")
    if summary["default_core_input_count"] != 35:
        raise ValueError("Default core-model input count must remain 35")
    return frame


def registry_summary(frame: pd.DataFrame) -> dict:
    core_defaults = frame.loc[
        frame["role"].eq("core_model") & frame["default_input"].str.lower().eq("yes")
    ]
    return {
        "row_count": int(len(frame)),
        "entity_counts": {str(key): int(value) for key, value in frame.groupby("entity").size().to_dict().items()},
        "role_counts": {str(key): int(value) for key, value in frame.groupby("role").size().to_dict().items()},
        "default_core_input_count": int(len(core_defaults)),
    }
