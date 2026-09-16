"""Global, roof-stratified query selection for the multicity experiment."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RoofArchetype:
    """An interpretable target in global roof-feature percentile space."""

    name: str
    label: str
    targets: tuple[float, ...]

    def validate(self, feature_count: int) -> None:
        if not str(self.name).strip() or not str(self.label).strip():
            raise ValueError("Archetype name and label cannot be empty")
        if len(self.targets) != feature_count:
            raise ValueError(
                f"Archetype {self.name!r} has {len(self.targets)} targets; "
                f"expected {feature_count}"
            )
        if any(not np.isfinite(value) or not 0 <= value <= 1 for value in self.targets):
            raise ValueError("Archetype targets must be finite percentiles from 0 to 1")


DEFAULT_ROOF_ARCHETYPES = (
    RoofArchetype("near_flat", "Near-flat", (0.10, 0.10, 0.10)),
    RoofArchetype("typical", "Typical", (0.50, 0.50, 0.50)),
    RoofArchetype("steep", "Steep", (0.90, 0.50, 0.50)),
    RoofArchetype("high_relief", "High relief", (0.50, 0.90, 0.50)),
    RoofArchetype("heterogeneous", "Heterogeneous", (0.50, 0.50, 0.90)),
)


def _require_columns(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing query-selection columns: {missing}")


def _stable_tie_token(seed: int, city: str, block_id: str) -> str:
    payload = f"{int(seed)}|{city}|{block_id}".encode("utf-8")
    return hashlib.blake2b(payload, digest_size=8).hexdigest()


def _robust_scaled_values(
    frame: pd.DataFrame, features: Sequence[str]
) -> tuple[np.ndarray, np.ndarray]:
    values = frame.loc[:, features].to_numpy(dtype=float)
    medians = np.nanmedian(values, axis=0)
    first, third = np.nanquantile(values, [0.25, 0.75], axis=0)
    scales = third - first
    scales[(~np.isfinite(scales)) | (scales == 0)] = 1.0
    scaled = (values - medians) / scales
    central_distance = np.sqrt(np.mean(np.square(scaled), axis=1))
    return scaled, central_distance


def _separation_from_selected(
    vector: np.ndarray, selected_vectors: list[np.ndarray]
) -> float:
    if not selected_vectors:
        return float("nan")
    distances = [
        float(np.sqrt(np.mean(np.square(vector - selected))))
        for selected in selected_vectors
    ]
    return min(distances)


def build_global_query_set(
    frame: pd.DataFrame,
    general_features: Sequence[str],
    roof_features: Sequence[str],
    *,
    archetypes: Sequence[RoofArchetype] = DEFAULT_ROOF_ARCHETYPES,
    queries_per_category: int = 10,
    max_per_city_per_category: int = 3,
    min_general_separation: float = 0.55,
    general_central_quantile: float = 0.98,
    seed: int = 20260910,
) -> pd.DataFrame:
    """Select globally defined roof strata with form diversity and a soft city cap.

    Roof targets are defined on percentile ranks computed over the complete pooled
    dataset. Candidates are considered in target-distance order. A decreasing
    separation threshold prevents near-duplicate general forms without blocking
    a category when the pool is dense.
    """

    if queries_per_category <= 0:
        raise ValueError("queries_per_category must be positive")
    if max_per_city_per_category <= 0:
        raise ValueError("max_per_city_per_category must be positive")
    if min_general_separation < 0:
        raise ValueError("min_general_separation cannot be negative")
    if not 0 < general_central_quantile <= 1:
        raise ValueError("general_central_quantile must be in (0, 1]")
    if not archetypes:
        raise ValueError("At least one roof archetype is required")

    general_features = tuple(general_features)
    roof_features = tuple(roof_features)
    for archetype in archetypes:
        archetype.validate(len(roof_features))
    names = [archetype.name for archetype in archetypes]
    if len(names) != len(set(names)):
        raise ValueError("Archetype names must be unique")

    required = ("city", "block_id", *general_features, *roof_features)
    _require_columns(frame, required)
    working = frame.copy()
    for column in (*general_features, *roof_features):
        working[column] = pd.to_numeric(working[column], errors="coerce")
    finite = np.isfinite(
        working.loc[:, (*general_features, *roof_features)].to_numpy(dtype=float)
    ).all(axis=1)
    working = working.loc[finite].copy().reset_index(drop=True)
    if working.empty:
        raise ValueError("No complete rows are available for query selection")
    if working.duplicated(["city", "block_id"]).any():
        raise ValueError("Each city/block_id pair must be unique")

    percentile_columns = []
    for feature in roof_features:
        percentile_column = f"{feature}_percentile"
        working[percentile_column] = working[feature].rank(
            method="average", pct=True
        )
        percentile_columns.append(percentile_column)

    general_z, central_distance = _robust_scaled_values(working, general_features)
    working["general_central_distance"] = central_distance
    cutoff = float(np.quantile(central_distance, general_central_quantile))
    working["query_general_scope"] = central_distance <= cutoff
    working["_tie_token"] = [
        _stable_tie_token(seed, city, block_id)
        for city, block_id in working[["city", "block_id"]].astype(str).itertuples(
            index=False, name=None
        )
    ]
    general_vectors = {
        (str(row.city), str(row.block_id)): general_z[index]
        for index, row in enumerate(working.itertuples(index=False))
    }

    thresholds = []
    for value in (
        min_general_separation,
        min_general_separation * 0.67,
        min_general_separation * 0.33,
        0.0,
    ):
        value = float(value)
        if value not in thresholds:
            thresholds.append(value)

    globally_used: set[tuple[str, str]] = set()
    selections: list[dict] = []
    percentile_values = working[percentile_columns].to_numpy(dtype=float)
    for archetype in archetypes:
        category = working.copy()
        target = np.asarray(archetype.targets, dtype=float)
        category["target_distance"] = np.sqrt(
            np.mean(np.square(percentile_values - target), axis=1)
        )
        category = category.loc[category["query_general_scope"]].sort_values(
            ["target_distance", "_tie_token"], kind="mergesort"
        )
        chosen_keys: set[tuple[str, str]] = set()
        chosen_vectors: list[np.ndarray] = []
        city_counts: dict[str, int] = {}
        for threshold in thresholds:
            for row in category.itertuples(index=False):
                key = (str(row.city), str(row.block_id))
                if key in globally_used or key in chosen_keys:
                    continue
                city = str(row.city)
                if city_counts.get(city, 0) >= max_per_city_per_category:
                    continue
                vector = general_vectors[key]
                separation = _separation_from_selected(vector, chosen_vectors)
                if chosen_vectors and separation + 1e-12 < threshold:
                    continue
                record = row._asdict()
                record.update(
                    {
                        "query_category": archetype.name,
                        "query_category_label": archetype.label,
                        "selection_rank": len(chosen_keys) + 1,
                        "general_separation_at_selection": separation,
                        "selection_threshold": threshold,
                    }
                )
                for feature, value in zip(roof_features, archetype.targets):
                    record[f"{feature}_target_percentile"] = float(value)
                selections.append(record)
                chosen_keys.add(key)
                globally_used.add(key)
                chosen_vectors.append(vector)
                city_counts[city] = city_counts.get(city, 0) + 1
                if len(chosen_keys) == queries_per_category:
                    break
            if len(chosen_keys) == queries_per_category:
                break
        if len(chosen_keys) != queries_per_category:
            raise ValueError(
                f"Could select only {len(chosen_keys)} of {queries_per_category} "
                f"queries for category {archetype.name!r} under the city cap"
            )

    result = pd.DataFrame(selections)
    result = result.drop(columns=["_tie_token"], errors="ignore")
    category_order = {archetype.name: index for index, archetype in enumerate(archetypes)}
    result["_category_order"] = result["query_category"].map(category_order)
    result = result.sort_values(
        ["_category_order", "selection_rank"], kind="mergesort"
    ).drop(columns="_category_order")
    return result.reset_index(drop=True)


def summarize_global_query_set(queries: pd.DataFrame) -> pd.DataFrame:
    """Return a compact coverage audit for a frozen global query set."""

    _require_columns(
        queries,
        (
            "query_category",
            "query_category_label",
            "city",
            "target_distance",
            "general_central_distance",
        ),
    )
    return (
        queries.groupby(
            ["query_category", "query_category_label"],
            sort=False,
            as_index=False,
        )
        .agg(
            query_count=("block_id", "size"),
            city_count=("city", "nunique"),
            max_city_count=("city", lambda values: int(values.value_counts().max())),
            median_target_distance=("target_distance", "median"),
            max_target_distance=("target_distance", "max"),
            median_general_central_distance=("general_central_distance", "median"),
        )
    )

