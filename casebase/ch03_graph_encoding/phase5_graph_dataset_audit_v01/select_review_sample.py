"""Deterministically select two visual-review Graphs per city."""

from __future__ import annotations

import pandas as pd


EXTREME_FIELDS = [
    "node_count", "connected_components", "isolated_node_ratio",
    "combined_weak_candidate_ratio", "excluded_building_ratio",
]


def select_review_cases(index: pd.DataFrame) -> pd.DataFrame:
    completed = index.loc[index["case_status"].eq("completed")].copy()
    selections: list[dict] = []
    for city, cases in completed.groupby("city", sort=True):
        if len(cases) < 2:
            raise ValueError(f"City {city} has fewer than two completed Graphs")
        cases = cases.copy()
        for field in EXTREME_FIELDS:
            cases[field] = pd.to_numeric(cases[field], errors="coerce")
        median = cases["node_count"].median()
        median_row = cases.assign(_distance=(cases["node_count"] - median).abs()).sort_values(
            ["_distance", "block_id", "sample_id"], kind="mergesort"
        ).iloc[0]
        thresholds = {field: cases[field].quantile(0.99) for field in EXTREME_FIELDS}
        hits = pd.DataFrame({field: cases[field].ge(value) for field, value in thresholds.items()}, index=cases.index)
        ranked = cases.assign(_hits=hits.sum(axis=1)).sort_values(
            ["_hits", "combined_weak_candidate_ratio", "node_count", "block_id", "sample_id"],
            ascending=[False, False, False, True, True], kind="mergesort",
        )
        extreme_row = ranked.loc[ranked["sample_id"].astype(str).ne(str(median_row["sample_id"]))].iloc[0]
        for source, chosen in (("city_median", median_row), ("city_structural_extreme", extreme_row)):
            item = chosen.drop(labels=[key for key in chosen.index if str(key).startswith("_")]).to_dict()
            item["selection_source"] = source
            item["selection_reason"] = (
                f"node_count nearest city median {median:g}" if source == "city_median"
                else ";".join(field for field in EXTREME_FIELDS if pd.notna(chosen[field]) and chosen[field] >= thresholds[field])
            )
            for field, value in thresholds.items():
                item[f"city_p99_{field}"] = value
            selections.append(item)
    result = pd.DataFrame(selections)
    return result.sort_values(["city", "selection_source", "block_id"], kind="mergesort").reset_index(drop=True)
