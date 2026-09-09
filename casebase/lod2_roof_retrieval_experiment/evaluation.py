"""Formal-query selection and non-scoring roof evaluation."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance


ROOF_FEATURES = (
    "roof_mean_slope",
    "roof_relief_ratio_median",
    "building_roof_slope_sd",
)

EVALUATION_FIELDS = (
    "d_general",
    "d_roof",
    "slope_wasserstein_deg",
    "relief_wasserstein",
    "flat_roof_share_abs_diff",
)


def _complete_usable_blocks(frame: pd.DataFrame) -> pd.DataFrame:
    required = ["block_id", *ROOF_FEATURES]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing block columns: {missing}")
    result = frame.copy()
    if "is_usable" in result.columns:
        result = result[result["is_usable"].fillna(False).astype(bool)]
    result = result.dropna(subset=required).copy()
    result["block_id"] = result["block_id"].astype(str)
    if result["block_id"].duplicated().any():
        raise ValueError("block_id values must be unique")
    return result


def select_formal_queries(
    blocks: pd.DataFrame,
    development_ids: Iterable[str],
    seed: int = 20260908,
    regular_count: int = 16,
) -> pd.DataFrame:
    """Select slope-stratified regular queries plus four boundary cases.

    Boundary cases are selected first and kept separate from the regular sample.
    The regular sample contains equal counts from four rank-based slope quartiles.
    """

    if regular_count <= 0 or regular_count % 4:
        raise ValueError("regular_count must be positive and divisible by four")
    excluded = {str(value) for value in development_ids}
    candidates = _complete_usable_blocks(blocks)
    candidates = candidates[~candidates["block_id"].isin(excluded)].copy()
    if len(candidates) < regular_count + 4:
        raise ValueError("Not enough non-development blocks for formal queries")

    boundary_specs = (
        ("near_flat", "roof_mean_slope", True),
        ("steep", "roof_mean_slope", False),
        ("high_relief", "roof_relief_ratio_median", False),
        ("high_dispersion", "building_roof_slope_sd", False),
    )
    used: set[str] = set()
    boundary_rows = []
    for stratum, feature, ascending in boundary_specs:
        available = candidates[~candidates["block_id"].isin(used)]
        ranked = available.sort_values(
            [feature, "block_id"],
            ascending=[ascending, True],
            kind="mergesort",
        )
        selected = ranked.iloc[0]
        block_id = str(selected["block_id"])
        used.add(block_id)
        boundary_rows.append(
            {
                "block_id": block_id,
                "query_role": "boundary",
                "stratum": stratum,
                **{feature_name: float(selected[feature_name]) for feature_name in ROOF_FEATURES},
            }
        )

    remaining = candidates[~candidates["block_id"].isin(used)].copy()
    remaining = remaining.sort_values("block_id", kind="mergesort")
    slope_rank = remaining["roof_mean_slope"].rank(method="first")
    remaining["slope_quartile"] = pd.qcut(slope_rank, 4, labels=False)
    per_quartile = regular_count // 4
    regular_rows = []
    for quartile in range(4):
        stratum_frame = remaining[remaining["slope_quartile"] == quartile]
        if len(stratum_frame) < per_quartile:
            raise ValueError(f"Slope quartile {quartile + 1} has too few blocks")
        sampled = stratum_frame.sample(
            n=per_quartile, random_state=int(seed) + quartile
        ).sort_values("block_id", kind="mergesort")
        for selected in sampled.to_dict("records"):
            regular_rows.append(
                {
                    "block_id": str(selected["block_id"]),
                    "query_role": "regular",
                    "stratum": f"slope_q{quartile + 1}",
                    **{
                        feature_name: float(selected[feature_name])
                        for feature_name in ROOF_FEATURES
                    },
                }
            )

    columns = ["block_id", "query_role", "stratum", *ROOF_FEATURES]
    return pd.DataFrame(boundary_rows + regular_rows, columns=columns)


def _building_distributions(buildings: pd.DataFrame) -> dict[str, dict[str, np.ndarray]]:
    required = ["block_id", "roof_mean_slope", "roof_relief_ratio"]
    missing = [column for column in required if column not in buildings.columns]
    if missing:
        raise ValueError(f"Missing building columns: {missing}")
    frame = buildings.copy()
    if "is_usable" in frame.columns:
        frame = frame[frame["is_usable"].fillna(False).astype(bool)]
    frame["block_id"] = frame["block_id"].astype(str)

    result: dict[str, dict[str, np.ndarray]] = {}
    for block_id, group in frame.groupby("block_id", sort=False):
        slopes = group["roof_mean_slope"].dropna().to_numpy(dtype=float)
        relief = group["roof_relief_ratio"].dropna().to_numpy(dtype=float)
        slopes = slopes[np.isfinite(slopes)]
        relief = relief[np.isfinite(relief)]
        if slopes.size and relief.size:
            result[str(block_id)] = {"slope": slopes, "relief": relief}
    return result


def evaluate_roof_distributions(
    buildings: pd.DataFrame, retrieval: pd.DataFrame
) -> pd.DataFrame:
    """Append building-level roof-distribution differences to retrieval rows."""

    required = ["query_id", "candidate_id"]
    missing_columns = [column for column in required if column not in retrieval.columns]
    if missing_columns:
        raise ValueError(f"Missing retrieval columns: {missing_columns}")
    distributions = _building_distributions(buildings)
    referenced = set(retrieval["query_id"].astype(str)) | set(
        retrieval["candidate_id"].astype(str)
    )
    missing_blocks = sorted(referenced - set(distributions))
    if missing_blocks:
        raise ValueError(
            "Missing usable building distributions for block(s): "
            + ", ".join(missing_blocks)
        )

    enriched = retrieval.copy()
    rows = []
    for row in enriched.itertuples(index=False):
        query = distributions[str(row.query_id)]
        candidate = distributions[str(row.candidate_id)]
        rows.append(
            {
                "slope_wasserstein_deg": float(
                    wasserstein_distance(query["slope"], candidate["slope"])
                ),
                "relief_wasserstein": float(
                    wasserstein_distance(query["relief"], candidate["relief"])
                ),
                "flat_roof_share_abs_diff": float(
                    abs(
                        np.mean(query["slope"] < 5.0)
                        - np.mean(candidate["slope"] < 5.0)
                    )
                ),
            }
        )
    return pd.concat([enriched.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def summarize_formal_evaluation(
    enriched: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return query means and aggregate raw/paired changes without a composite score."""

    missing = [field for field in EVALUATION_FIELDS if field not in enriched.columns]
    if missing:
        raise ValueError(f"Missing evaluation fields: {missing}")
    group_fields = ["query_id", "method", "lambda_value"]
    query_summary = (
        enriched.groupby(group_fields, as_index=False)[list(EVALUATION_FIELDS)]
        .mean()
        .sort_values(group_fields, kind="mergesort")
        .reset_index(drop=True)
    )
    baseline = (
        query_summary[query_summary["method"] == "M1"]
        .drop_duplicates("query_id")
        .set_index("query_id")
    )
    missing_baselines = sorted(set(query_summary["query_id"]) - set(baseline.index))
    if missing_baselines:
        raise ValueError(f"Missing M1 baseline for query(s): {missing_baselines}")
    for field in EVALUATION_FIELDS:
        query_summary[f"{field}_delta_vs_m1"] = query_summary.apply(
            lambda row: float(row[field] - baseline.loc[row["query_id"], field]), axis=1
        )

    aggregate_rows = []
    delta_fields = [f"{field}_delta_vs_m1" for field in EVALUATION_FIELDS]
    for (method, lambda_value), group in query_summary.groupby(
        ["method", "lambda_value"], sort=True
    ):
        row = {
            "method": method,
            "lambda_value": float(lambda_value),
            "query_count": int(group["query_id"].nunique()),
        }
        for field in [*EVALUATION_FIELDS, *delta_fields]:
            row[f"{field}_mean"] = float(group[field].mean())
            row[f"{field}_median"] = float(group[field].median())
        aggregate_rows.append(row)
    return query_summary, pd.DataFrame(aggregate_rows)
