"""Deterministic paired retrieval for the Luxembourg roof experiment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RobustFeatureScaler:
    """Median/IQR scaling whose zero-IQR features receive a safe scale of one."""

    features: tuple[str, ...]
    medians: pd.Series
    iqrs: pd.Series

    @classmethod
    def fit(cls, frame: pd.DataFrame, features: Sequence[str]):
        feature_names = tuple(features)
        values = frame.loc[:, feature_names].astype(float)
        medians = values.median()
        iqrs = values.quantile(0.75) - values.quantile(0.25)
        iqrs = iqrs.mask((~np.isfinite(iqrs)) | (iqrs == 0), 1.0)
        return cls(feature_names, medians, iqrs)

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        values = frame.loc[:, self.features].astype(float)
        return (values - self.medians) / self.iqrs


def _rmse_from_query(values: np.ndarray, query: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean((values - query) ** 2, axis=1))


def retrieve_for_query(
    frame: pd.DataFrame,
    query_id: str,
    general_features: Sequence[str],
    roof_features: Sequence[str],
    pool_size: int = 50,
    top_k: int = 5,
    lambda_values: Iterable[float] = (0.20, 0.35, 0.50),
    general_scaler: RobustFeatureScaler | None = None,
    roof_scaler: RobustFeatureScaler | None = None,
) -> pd.DataFrame:
    """Run M1 and roof-reranked M2 against one shared general-form pool."""

    if query_id not in set(frame["block_id"].astype(str)):
        raise KeyError(f"Unknown query block: {query_id}")
    if top_k <= 0 or pool_size <= 0:
        raise ValueError("pool_size and top_k must be positive")
    if top_k > pool_size:
        raise ValueError("top_k cannot exceed pool_size")

    general_scaler = general_scaler or RobustFeatureScaler.fit(frame, general_features)
    roof_scaler = roof_scaler or RobustFeatureScaler.fit(frame, roof_features)
    general_z = general_scaler.transform(frame)
    roof_z = roof_scaler.transform(frame)
    query_position = frame.index[frame["block_id"].astype(str) == str(query_id)][0]
    query_general = general_z.loc[query_position].to_numpy(dtype=float)
    query_roof = roof_z.loc[query_position].to_numpy(dtype=float)

    candidates = frame.loc[frame.index != query_position, ["block_id"]].copy()
    candidates["candidate_id"] = candidates["block_id"].astype(str)
    candidates["d_general"] = _rmse_from_query(
        general_z.loc[candidates.index].to_numpy(dtype=float), query_general
    )
    candidates["d_roof"] = _rmse_from_query(
        roof_z.loc[candidates.index].to_numpy(dtype=float), query_roof
    )
    candidates = candidates.sort_values(
        ["d_general", "candidate_id"], kind="mergesort"
    ).head(pool_size)
    candidates["general_pool_rank"] = np.arange(1, len(candidates) + 1)
    actual_pool_size = len(candidates)
    if actual_pool_size < top_k:
        raise ValueError("Not enough non-query candidates for requested top_k")

    rows = []

    def append_ranked(ranked: pd.DataFrame, method: str, lambda_value: float) -> None:
        for rank, candidate in enumerate(ranked.itertuples(index=False), start=1):
            rows.append(
                {
                    "query_id": str(query_id),
                    "method": method,
                    "lambda_value": float(lambda_value),
                    "rank": rank,
                    "candidate_id": candidate.candidate_id,
                    "d_general": float(candidate.d_general),
                    "d_roof": float(candidate.d_roof),
                    "d_combined": float(candidate.d_combined),
                    "pool_size": actual_pool_size,
                    "general_pool_rank": int(candidate.general_pool_rank),
                }
            )

    baseline = candidates.copy()
    baseline["d_combined"] = baseline["d_general"]
    append_ranked(baseline.head(top_k), "M1", 0.0)

    for lambda_value in lambda_values:
        lambda_value = float(lambda_value)
        if not 0 <= lambda_value <= 1:
            raise ValueError("lambda values must be between zero and one")
        reranked = candidates.copy()
        reranked["d_combined"] = (
            (1.0 - lambda_value) * reranked["d_general"]
            + lambda_value * reranked["d_roof"]
        )
        reranked = reranked.sort_values(
            ["d_combined", "candidate_id"], kind="mergesort"
        )
        append_ranked(reranked.head(top_k), "M2", lambda_value)
    return pd.DataFrame(rows)


def select_development_queries(frame: pd.DataFrame) -> pd.DataFrame:
    """Choose five distinct, interpretable roof-feature development queries."""

    required = [
        "block_id",
        "roof_mean_slope",
        "building_roof_slope_sd",
        "roof_relief_ratio_median",
    ]
    candidates = frame.loc[:, required].dropna().copy()
    if len(candidates) < 5:
        raise ValueError("At least five complete blocks are required")

    slope = candidates["roof_mean_slope"]
    dispersion = candidates["building_roof_slope_sd"]
    relief = candidates["roof_relief_ratio_median"]
    target_specs = [
        ("lowest_mean_slope", {"roof_mean_slope": float(slope.min())}),
        ("highest_mean_slope", {"roof_mean_slope": float(slope.max())}),
        (
            "median_slope_low_dispersion",
            {
                "roof_mean_slope": float(slope.median()),
                "building_roof_slope_sd": float(dispersion.min()),
            },
        ),
        (
            "median_slope_high_dispersion",
            {
                "roof_mean_slope": float(slope.median()),
                "building_roof_slope_sd": float(dispersion.max()),
            },
        ),
        ("highest_relief_ratio", {"roof_relief_ratio_median": float(relief.max())}),
    ]

    scales = {}
    for feature in required[1:]:
        iqr = candidates[feature].quantile(0.75) - candidates[feature].quantile(0.25)
        scales[feature] = float(iqr) if np.isfinite(iqr) and iqr > 0 else 1.0

    used: set[str] = set()
    selections = []
    for query_type, targets in target_specs:
        ranked = candidates.copy()
        squared = np.zeros(len(ranked), dtype=float)
        for feature, target in targets.items():
            squared += ((ranked[feature].to_numpy(dtype=float) - target) / scales[feature]) ** 2
        ranked["target_distance"] = np.sqrt(squared / len(targets))
        ranked["block_id"] = ranked["block_id"].astype(str)
        ranked = ranked.sort_values(["target_distance", "block_id"], kind="mergesort")
        selected = ranked.loc[~ranked["block_id"].isin(used)].iloc[0]
        used.add(selected["block_id"])
        selections.append(
            {
                "query_type": query_type,
                "block_id": selected["block_id"],
                "target_distance": float(selected["target_distance"]),
                "roof_mean_slope": float(selected["roof_mean_slope"]),
                "building_roof_slope_sd": float(selected["building_roof_slope_sd"]),
                "roof_relief_ratio_median": float(selected["roof_relief_ratio_median"]),
            }
        )
    return pd.DataFrame(selections)


def run_development_retrieval(
    frame: pd.DataFrame,
    general_features: Sequence[str],
    roof_features: Sequence[str],
    pool_size: int = 50,
    top_k: int = 5,
    lambda_values: Iterable[float] = (0.20, 0.35, 0.50),
    queries: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, RobustFeatureScaler, RobustFeatureScaler]:
    """Select development queries and concatenate their paired retrieval results."""

    queries = select_development_queries(frame) if queries is None else queries.copy()
    general_scaler = RobustFeatureScaler.fit(frame, general_features)
    roof_scaler = RobustFeatureScaler.fit(frame, roof_features)
    results = [
        retrieve_for_query(
            frame,
            query_id,
            general_features,
            roof_features,
            pool_size,
            top_k,
            lambda_values,
            general_scaler,
            roof_scaler,
        )
        for query_id in queries["block_id"]
    ]
    return queries, pd.concat(results, ignore_index=True), general_scaler, roof_scaler
