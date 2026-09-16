import unittest

import numpy as np
import pandas as pd

try:
    from global_query_selection import RoofArchetype, build_global_query_set
except (ImportError, ModuleNotFoundError):
    RoofArchetype = None
    build_global_query_set = None


ROOF_FEATURES = ("slope", "relief", "dispersion")
GENERAL_FEATURES = ("area", "density")


def selection_fixture() -> pd.DataFrame:
    rows = []
    for city_index, city in enumerate(("a", "b", "c", "d")):
        for value_index, value in enumerate(np.linspace(0.0, 1.0, 8)):
            rows.append(
                {
                    "city": city,
                    "block_id": f"{city}-{value_index}",
                    "slope": value,
                    "relief": value,
                    "dispersion": value,
                    "area": value_index * 2.0 + city_index * 0.1,
                    "density": (7 - value_index) * 1.5 + city_index * 0.1,
                }
            )
    return pd.DataFrame(rows)


class GlobalQuerySelectionTests(unittest.TestCase):
    def test_global_archetypes_select_unique_queries_with_a_per_category_city_cap(self):
        self.assertIsNotNone(build_global_query_set)
        archetypes = (
            RoofArchetype("low", "Low", (0.10, 0.10, 0.10)),
            RoofArchetype("high", "High", (0.90, 0.90, 0.90)),
        )

        selected = build_global_query_set(
            selection_fixture(),
            general_features=GENERAL_FEATURES,
            roof_features=ROOF_FEATURES,
            archetypes=archetypes,
            queries_per_category=3,
            max_per_city_per_category=1,
            min_general_separation=0.25,
            general_central_quantile=1.0,
            seed=20260910,
        )

        self.assertEqual(selected.groupby("query_category").size().to_dict(), {"high": 3, "low": 3})
        self.assertEqual(selected[["city", "block_id"]].drop_duplicates().shape[0], 6)
        self.assertLessEqual(
            int(selected.groupby(["query_category", "city"]).size().max()), 1
        )
        self.assertTrue(selected["target_distance"].between(0, 1).all())
        self.assertEqual(
            set(selected.columns) & {"slope_percentile", "relief_percentile", "dispersion_percentile"},
            {"slope_percentile", "relief_percentile", "dispersion_percentile"},
        )

    def test_impossible_city_cap_fails_instead_of_silently_returning_too_few(self):
        one_city = selection_fixture().query("city == 'a'")
        archetypes = (RoofArchetype("typical", "Typical", (0.5, 0.5, 0.5)),)

        with self.assertRaisesRegex(ValueError, "Could select only"):
            build_global_query_set(
                one_city,
                general_features=GENERAL_FEATURES,
                roof_features=ROOF_FEATURES,
                archetypes=archetypes,
                queries_per_category=2,
                max_per_city_per_category=1,
                general_central_quantile=1.0,
            )

    def test_incomplete_rows_and_general_outliers_are_not_selected(self):
        frame = selection_fixture()
        frame.loc[0, "slope"] = np.nan
        outlier = frame.iloc[[1]].copy()
        outlier["block_id"] = "outlier"
        outlier[["area", "density"]] = 1_000_000.0
        frame = pd.concat([frame, outlier], ignore_index=True)
        archetypes = (RoofArchetype("low", "Low", (0.10, 0.10, 0.10)),)

        selected = build_global_query_set(
            frame,
            general_features=GENERAL_FEATURES,
            roof_features=ROOF_FEATURES,
            archetypes=archetypes,
            queries_per_category=3,
            max_per_city_per_category=2,
            general_central_quantile=0.95,
        )

        self.assertNotIn("a-0", set(selected["block_id"]))
        self.assertNotIn("outlier", set(selected["block_id"]))


if __name__ == "__main__":
    unittest.main()
