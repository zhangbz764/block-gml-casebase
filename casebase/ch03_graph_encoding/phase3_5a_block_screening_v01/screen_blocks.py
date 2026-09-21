"""Pure, deterministic Phase 3.5A block-screening rules."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PHASE_DIR = Path(__file__).resolve().parent
RULES = json.loads((PHASE_DIR / "screening_rules_v0.1.json").read_text(encoding="utf-8"))
METRIC_FIELDS = tuple(RULES["metric_fields"])
EXPECTED_SINGLE_NULLS = frozenset(RULES["single_building_expected_nulls"])
HARD_RULES = tuple(RULES["hard_rule_priority"])
SCOPE_RULES = tuple(RULES["scope_rule_names"])
GEOMETRY_FIELDS = (
    "geom_is_null",
    "geom_is_empty",
    "geom_is_valid",
    "geom_area_positive",
)
SCOPE_METRICS = (
    "area_m2",
    "elongation",
    "compactness",
    "lod1_building_count",
    "bcr",
)


def validate_candidate_frame(frame: pd.DataFrame) -> None:
    """Validate required fields and metric types without mutating the input."""
    required = {"city", "block_id", *METRIC_FIELDS, *GEOMETRY_FIELDS}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Candidate frame is missing required fields: {missing}")
    if frame.empty:
        raise ValueError("Candidate frame must contain at least one row")
    for field in METRIC_FIELDS:
        converted = pd.to_numeric(frame[field], errors="coerce")
        invalid = frame[field].notna() & converted.isna()
        if invalid.any():
            raise TypeError(f"Metric field {field} contains non-numeric values")


def _numeric_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[:, METRIC_FIELDS].apply(pd.to_numeric, errors="coerce")


def _join_true_reasons(flags: pd.DataFrame, names: Mapping[str, str], order: tuple[str, ...]) -> pd.Series:
    return flags.loc[:, order].apply(
        lambda row: ";".join(names[rule] for rule in order if bool(row[rule])),
        axis=1,
    )


def compute_hard_flags(frame: pd.DataFrame) -> pd.DataFrame:
    """Return H01-H05 flags, detailed QA failures and hard-rule reasons."""
    validate_candidate_frame(frame)
    numeric = _numeric_frame(frame)
    city_missing = frame["city"].isna() | frame["city"].astype("string").str.strip().eq("").fillna(True)
    block_missing = frame["block_id"].isna() | frame["block_id"].astype("string").str.strip().eq("").fillna(True)
    duplicate_key = frame.duplicated(["city", "block_id"], keep=False)
    geometry_invalid = (
        frame["geom_is_null"].fillna(True).astype(bool)
        | frame["geom_is_empty"].fillna(True).astype(bool)
        | ~frame["geom_is_valid"].fillna(False).astype(bool)
        | ~frame["geom_area_positive"].fillna(False).astype(bool)
    )
    h01 = city_missing | block_missing | duplicate_key | geometry_invalid

    qa_masks: dict[str, pd.Series] = {
        "nonpositive_block_area": numeric["area_m2"].le(0),
        "elongation_outside_0_1": numeric["elongation"].le(0) | numeric["elongation"].gt(1.000001),
        "compactness_outside_0_1": numeric["compactness"].le(0) | numeric["compactness"].gt(1.000001),
        "candidate_without_building": numeric["lod1_building_count"].lt(1),
        "negative_bcr": numeric["bcr"].lt(0),
        "convexity_outside_0_1": numeric["convexity"].le(0) | numeric["convexity"].gt(1.000001),
        "rectangularity_outside_0_1": numeric["rectangularity"].le(0) | numeric["rectangularity"].gt(1.000001),
        "negative_far": numeric["far"].lt(0),
        "nonpositive_bcd": numeric["bcd"].le(0),
        "nonpositive_average_height": numeric["avh"].le(0),
        "nonpositive_weighted_height": numeric["avh_w"].le(0),
        "negative_height_sd": numeric["hsd"].lt(0),
        "negative_height_cv": numeric["hcv"].lt(0),
        "nonpositive_max_height": numeric["h_max"].le(0),
        "negative_height_range": numeric["h_range"].lt(0),
        "nonpositive_total_volume": numeric["total_volume"].le(0),
        "nonpositive_average_volume": numeric["avg_volume"].le(0),
        "nonpositive_average_footprint": numeric["avg_footprint"].le(0),
        "negative_footprint_cv": numeric["footprint_cv"].lt(0),
        "complexity_below_1": numeric["avg_complexity"].lt(0.999999),
        "building_rectangularity_outside_0_1": numeric["avg_rectangularity"].le(0) | numeric["avg_rectangularity"].gt(1.000001),
        "building_convexity_outside_0_1": numeric["avg_convexity"].le(0) | numeric["avg_convexity"].gt(1.000001),
        "orientation_outside_0_180": numeric["avg_orientation"].lt(0) | numeric["avg_orientation"].ge(180),
        "orientation_entropy_outside_range": numeric["orientation_entropy"].lt(0) | numeric["orientation_entropy"].gt(math.log(18) + 1e-8),
        "negative_nn_distance": numeric["avg_nn_distance"].lt(0),
    }
    qa_masks["osr_formula_mismatch"] = (
        numeric["osr"].notna()
        & numeric["bcr"].notna()
        & numeric["far"].gt(0)
        & (
            (numeric["osr"] - (1.0 - numeric["bcr"]) / numeric["far"]).abs()
            > 1e-8 * np.maximum(1.0, numeric["osr"].abs())
        )
    )
    qa_masks["average_height_above_max"] = (
        numeric["avh"].notna() & numeric["h_max"].notna()
        & numeric["avh"].gt(numeric["h_max"] + 1e-8)
    )
    qa_masks["weighted_height_above_max"] = (
        numeric["avh_w"].notna() & numeric["h_max"].notna()
        & numeric["avh_w"].gt(numeric["h_max"] + 1e-8)
    )
    qa_masks["average_volume_above_total"] = (
        numeric["avg_volume"].notna() & numeric["total_volume"].notna()
        & numeric["avg_volume"].gt(numeric["total_volume"] + 1e-8)
    )
    qa_order = tuple(RULES["qa_error_rules"])
    qa_frame = pd.DataFrame({name: qa_masks[name].fillna(False).astype(bool) for name in qa_order}, index=frame.index)
    h02 = qa_frame.any(axis=1)
    h02_reasons = qa_frame.apply(
        lambda row: ";".join(name for name in qa_order if bool(row[name])), axis=1
    )

    h03 = numeric["bcr"].gt(1) | numeric["osr"].lt(0)
    missing_metrics = numeric.isna()
    unexpected_single = missing_metrics.loc[:, [field for field in METRIC_FIELDS if field not in EXPECTED_SINGLE_NULLS]].any(axis=1)
    h04 = (
        numeric["lod1_building_count"].eq(1) & unexpected_single
    ) | (
        numeric["lod1_building_count"].gt(1) & missing_metrics.any(axis=1)
    )
    h05 = numeric["lod1_building_count"].gt(1) & numeric["avg_nn_distance"].eq(0)

    result = pd.DataFrame(
        {
            "city": frame["city"],
            "block_id": frame["block_id"],
            "H01": h01.fillna(True).astype(bool),
            "H02": h02.fillna(False).astype(bool),
            "H03": h03.fillna(False).astype(bool),
            "H04": h04.fillna(False).astype(bool),
            "H05": h05.fillna(False).astype(bool),
            "H02_qa_reasons": h02_reasons,
        },
        index=frame.index,
    )
    result["hard_flag_count"] = result.loc[:, HARD_RULES].sum(axis=1).astype(int)
    result["hard_reasons"] = _join_true_reasons(result, RULES["hard_rule_names"], HARD_RULES)
    result["primary_hard_reason"] = result.loc[:, HARD_RULES].apply(
        lambda row: next((RULES["hard_rule_names"][rule] for rule in HARD_RULES if bool(row[rule])), ""),
        axis=1,
    )
    return result


def compute_review_thresholds(eligible: pd.DataFrame) -> dict[str, float]:
    """Compute review-only P01/P99 triggers from hard-rule-eligible rows."""
    if eligible.empty:
        return {
            "eligible_count": 0,
            "area_m2_p99": math.nan,
            "elongation_p01": math.nan,
            "compactness_p01": math.nan,
            "lod1_building_count_p99": math.nan,
            "bcr_p01": math.nan,
        }
    lower = float(RULES["scope_quantiles"]["lower"])
    upper = float(RULES["scope_quantiles"]["upper"])
    numeric = eligible.loc[:, SCOPE_METRICS].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any():
        raise ValueError("Review-threshold metrics must be complete in eligible rows")
    return {
        "eligible_count": int(len(eligible)),
        "area_m2_p99": float(numeric["area_m2"].quantile(upper)),
        "elongation_p01": float(numeric["elongation"].quantile(lower)),
        "compactness_p01": float(numeric["compactness"].quantile(lower)),
        "lod1_building_count_p99": float(numeric["lod1_building_count"].quantile(upper)),
        "bcr_p01": float(numeric["bcr"].quantile(lower)),
    }


def compute_scope_flags(eligible: pd.DataFrame, thresholds: Mapping[str, float]) -> pd.DataFrame:
    """Return S01-S06 and explanatory percentiles; never return a deletion decision."""
    numeric = eligible.loc[:, SCOPE_METRICS].apply(pd.to_numeric, errors="coerce")
    result = pd.DataFrame(
        {
            "city": eligible["city"],
            "block_id": eligible["block_id"],
            "S01": numeric["area_m2"].ge(thresholds["area_m2_p99"]),
            "S02": numeric["elongation"].le(thresholds["elongation_p01"]),
            "S03": numeric["compactness"].le(thresholds["compactness_p01"]),
            "S04": numeric["lod1_building_count"].ge(thresholds["lod1_building_count_p99"]),
            "S05": numeric["bcr"].le(thresholds["bcr_p01"]),
        },
        index=eligible.index,
    )
    if len(eligible) < 2:
        result.loc[:, ["S01", "S02", "S03", "S04", "S05"]] = False
    result["S06"] = result["S01"] & result["S05"]
    for field in SCOPE_METRICS:
        result[f"{field}_global_percentile"] = numeric[field].rank(method="average", pct=True)
        result[f"{field}_city_percentile"] = numeric.groupby(eligible["city"], sort=True)[field].rank(
            method="average", pct=True
        )
    result["scope_flag_count"] = result.loc[:, SCOPE_RULES].sum(axis=1).astype(int)
    result["scope_reasons"] = _join_true_reasons(result, RULES["scope_rule_names"], SCOPE_RULES)
    return result


def assign_machine_status(flagged: pd.DataFrame) -> pd.DataFrame:
    """Assign one status with hard exclusion taking precedence over review scope."""
    result = flagged.copy()
    result["machine_status"] = np.select(
        [result["hard_flag_count"].gt(0), result["scope_flag_count"].gt(0)],
        ["exclude_hard", "review_scope"],
        default="keep_provisional",
    )
    return result


def screen_dataframe(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply all v0.1 rules and return a stable, auditable ledger."""
    validate_candidate_frame(frame)
    ordered = frame.copy().sort_values(["city", "block_id"], kind="mergesort", na_position="first").reset_index(drop=True)
    hard = compute_hard_flags(ordered)
    ledger = pd.concat([ordered, hard.drop(columns=["city", "block_id"])], axis=1)
    eligible_mask = ledger["hard_flag_count"].eq(0)
    thresholds = compute_review_thresholds(ledger.loc[eligible_mask])

    percentile_columns = [
        f"{field}_{scope}_percentile"
        for field in SCOPE_METRICS
        for scope in ("global", "city")
    ]
    for rule in SCOPE_RULES:
        ledger[rule] = False
    for column in percentile_columns:
        ledger[column] = np.nan
    ledger["scope_flag_count"] = 0
    ledger["scope_reasons"] = ""
    scope = compute_scope_flags(ledger.loc[eligible_mask], thresholds)
    scope_columns = [*SCOPE_RULES, *percentile_columns, "scope_flag_count", "scope_reasons"]
    ledger.loc[eligible_mask, scope_columns] = scope.loc[:, scope_columns]
    for rule in SCOPE_RULES:
        ledger[rule] = ledger[rule].astype(bool)
    ledger["scope_flag_count"] = ledger["scope_flag_count"].astype(int)
    ledger = assign_machine_status(ledger)
    ledger.insert(2, "source_relation", RULES["source_relation"])
    ledger.insert(3, "rule_version", RULES["rule_version"])
    for field in ("manual_decision", "manual_reason", "manual_notes", "reviewed_at"):
        ledger[field] = ""
    ledger.attrs["thresholds"] = thresholds
    return ledger


def build_filter_summaries(ledger: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build deterministic sequential, independent, overlap and city summaries."""
    remaining = pd.Series(True, index=ledger.index)
    cumulative = 0
    flow_rows: list[dict[str, Any]] = []
    for rule in HARD_RULES:
        newly = remaining & ledger[rule].astype(bool)
        new_count = int(newly.sum())
        cumulative += new_count
        remaining &= ~ledger[rule].astype(bool)
        flow_rows.append(
            {
                "rule_id": rule,
                "rule_name": RULES["hard_rule_names"][rule],
                "input_count": len(ledger),
                "newly_excluded_count": new_count,
                "cumulative_excluded_count": cumulative,
                "remaining_count": int(remaining.sum()),
            }
        )
    flow = pd.DataFrame(flow_rows)

    reason_rows = [
        {"rule_id": rule, "rule_name": RULES["hard_rule_names"][rule], "rule_type": "hard", "matched_count": int(ledger[rule].sum())}
        for rule in HARD_RULES
    ] + [
        {"rule_id": rule, "rule_name": RULES["scope_rule_names"][rule], "rule_type": "scope", "matched_count": int(ledger[rule].sum())}
        for rule in SCOPE_RULES
    ]
    reasons = pd.DataFrame(reason_rows)

    overlap = (
        ledger.assign(
            rule_overlap=ledger.apply(
                lambda row: ";".join(
                    [rule for rule in (*HARD_RULES, *SCOPE_RULES) if bool(row[rule])]
                ) or "none",
                axis=1,
            )
        )
        .groupby(["machine_status", "rule_overlap"], dropna=False, sort=True)
        .size()
        .rename("block_count")
        .reset_index()
    )

    city_source = ledger.assign(city=ledger["city"].where(ledger["city"].notna(), "(missing)"))
    city = city_source.groupby("city", sort=True).size().rename("input_count").to_frame()
    status_counts = pd.crosstab(city_source["city"], city_source["machine_status"])
    for status in ("exclude_hard", "review_scope", "keep_provisional"):
        city[f"{status}_count"] = status_counts.get(status, 0)
        city[f"{status}_ratio"] = city[f"{status}_count"] / city["input_count"]
    city = city.reset_index()
    return {
        "filter_flow_summary": flow,
        "filter_reason_summary": reasons,
        "filter_overlap_summary": overlap,
        "filter_summary_by_city": city,
    }
