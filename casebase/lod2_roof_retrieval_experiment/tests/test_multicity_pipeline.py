import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from block_metrics import BUILDING_METRIC_COLUMNS

try:
    from multicity_pipeline import (
        FROZEN_M1_FEATURES,
        collect_city_sample,
        classify_block_quality,
        config_fingerprint,
        id_set_fingerprint,
        load_city_checkpoint,
        randomized_block_ids,
        stable_city_seed,
        take_first_eligible_in_order,
        write_city_checkpoint,
    )
except (ImportError, ModuleNotFoundError):
    FROZEN_M1_FEATURES = None
    collect_city_sample = None
    classify_block_quality = None
    config_fingerprint = None
    id_set_fingerprint = None
    randomized_block_ids = None
    stable_city_seed = None
    take_first_eligible_in_order = None
    load_city_checkpoint = None
    write_city_checkpoint = None


def _metric_row(block_id, *, bcr=0.20, upstream_usable=True):
    return {
        "block_id": block_id,
        "area_m2": 10_000.0,
        "compactness": 0.70,
        "elongation": 0.60,
        "bcr": bcr,
        "volume_density": 1.5,
        "mean_h_g1": 12.0,
        "h_g1_cv": 0.20,
        "log_block_area": np.log(10_000.0),
        "log_building_count": np.log(4.0),
        "mean_footprint_area": 500.0,
        "footprint_area_cv": 0.30,
        "orientation_entropy": 0.40,
        "normalized_nn_distance": 0.10,
        "roof_mean_slope": 25.0,
        "roof_relief_ratio_median": 0.25,
        "building_roof_slope_sd": 6.0,
        "is_usable": upstream_usable,
        "quality_codes": "" if upstream_usable else "coverage_below_80pct",
    }


class RandomSelectionTests(unittest.TestCase):
    def test_checkpoint_is_marker_last_and_corruption_falls_back_to_recompute(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {
                "manifest": root / "manifest.parquet",
                "processed_blocks": root / "processed.parquet",
                "selected_blocks": root / "selected.parquet",
                "selected_buildings": root / "buildings.parquet",
                "metadata": root / "metadata.json",
            }
            frames = {
                "manifest": pd.DataFrame(columns=[
                    "block_id", "random_order", "processed", "data_valid",
                    "in_reference_scope", "is_eligible", "selected",
                    "exclusion_codes",
                ]),
                "processed_blocks": collect_city_sample([], lambda _: None).processed_blocks,
                "selected_blocks": collect_city_sample([], lambda _: None).selected_blocks,
                "selected_buildings": pd.DataFrame(columns=BUILDING_METRIC_COLUMNS),
            }

            write_city_checkpoint(paths, frames, {"fingerprint": "abc"})
            loaded = load_city_checkpoint(paths, "abc")

            self.assertIsNotNone(loaded)
            self.assertEqual(set(loaded), set(frames))
            self.assertFalse(any(root.glob("*.tmp*")))

            paths["metadata"].write_text("not-json", encoding="utf-8")
            self.assertIsNone(load_city_checkpoint(paths, "abc"))

    def test_empty_or_fully_rejected_samples_keep_stable_output_schemas(self):
        empty = collect_city_sample([], lambda _: None)
        self.assertIn("processed", empty.manifest.columns)
        self.assertIn("data_valid", empty.processed_blocks.columns)
        self.assertEqual(
            empty.selected_buildings.columns.tolist(), BUILDING_METRIC_COLUMNS
        )

        rejected = collect_city_sample(
            ["a"],
            lambda _: (
                pd.DataFrame([_metric_row("a", upstream_usable=False)]),
                pd.DataFrame(columns=BUILDING_METRIC_COLUMNS),
            ),
            target_count=1,
        )
        self.assertTrue(rejected.selected_blocks.empty)
        self.assertEqual(
            rejected.selected_buildings.columns.tolist(), BUILDING_METRIC_COLUMNS
        )

    def test_config_fingerprint_is_order_stable_and_change_sensitive(self):
        self.assertIsNotNone(config_fingerprint)
        first = config_fingerprint({"target": 5000, "cities": ["a", "b"]})
        reordered = config_fingerprint({"cities": ["a", "b"], "target": 5000})
        changed = config_fingerprint({"target": 4000, "cities": ["a", "b"]})

        self.assertEqual(first, reordered)
        self.assertNotEqual(first, changed)
        self.assertEqual(len(first), 16)

        ids = id_set_fingerprint(["b", "a", "a"])
        self.assertEqual(ids, id_set_fingerprint(["a", "b"]))
        self.assertNotEqual(ids, id_set_fingerprint(["a", "c"]))

    def test_city_order_is_repeatable_and_input_order_independent(self):
        self.assertIsNotNone(randomized_block_ids)
        ids = ["b3", "b1", "b4", "b2"]

        first = randomized_block_ids(ids, base_seed=20260909, city="berlin")
        repeated = randomized_block_ids(reversed(ids), base_seed=20260909, city="Berlin")
        other_city = randomized_block_ids(ids, base_seed=20260909, city="vienna")

        self.assertEqual(first, repeated)
        self.assertEqual(set(first), set(ids))
        self.assertNotEqual(first, other_city)
        self.assertEqual(
            stable_city_seed(20260909, "Berlin"),
            stable_city_seed(20260909, "berlin"),
        )

    def test_takes_first_eligible_blocks_in_randomized_order(self):
        self.assertIsNotNone(take_first_eligible_in_order)
        frame = pd.DataFrame(
            {
                "block_id": ["a", "b", "c", "d"],
                "is_eligible": [True, True, False, True],
            }
        )

        selected = take_first_eligible_in_order(
            frame, ordered_ids=["c", "d", "a", "b"], target_count=2
        )

        self.assertEqual(selected["block_id"].tolist(), ["d", "a"])

    def test_city_sampler_processes_batches_until_target_and_keeps_manifest(self):
        self.assertIsNotNone(collect_city_sample)

        def measure_batch(block_ids):
            bcr = {"a": 0.20, "b": 0.01, "c": 1.01, "d": 0.30, "e": 0.40}
            blocks = pd.DataFrame([_metric_row(block_id, bcr=bcr[block_id]) for block_id in block_ids])
            buildings = pd.DataFrame(
                {"building_id": [f"building-{value}" for value in block_ids], "block_id": block_ids}
            )
            return blocks, buildings

        result = collect_city_sample(
            ordered_ids=["a", "b", "c", "d", "e"],
            measure_batch=measure_batch,
            target_count=2,
            batch_size=2,
            min_bcr=0.02,
        )

        self.assertEqual(result.selected_blocks["block_id"].tolist(), ["a", "d"])
        self.assertEqual(
            result.selected_buildings["building_id"].tolist(),
            ["building-a", "building-d"],
        )
        manifest = result.manifest.set_index("block_id")
        self.assertTrue(manifest.loc["a", "selected"])
        self.assertTrue(manifest.loc["d", "selected"])
        self.assertFalse(manifest.loc["e", "processed"])
        self.assertEqual(manifest.loc["b", "exclusion_codes"], "below_reference_bcr")


class HardFilterTests(unittest.TestCase):
    def test_separates_data_errors_from_low_bcr_scope_exclusion(self):
        self.assertIsNotNone(classify_block_quality)
        frame = pd.DataFrame(
            [
                _metric_row("valid"),
                _metric_row("low-bcr", bcr=0.01),
                _metric_row("over-one", bcr=1.01),
                _metric_row("low-coverage", upstream_usable=False),
            ]
        )

        result = classify_block_quality(frame, min_bcr=0.02).set_index("block_id")

        self.assertTrue(result.loc["valid", "data_valid"])
        self.assertTrue(result.loc["valid", "is_eligible"])
        self.assertTrue(result.loc["low-bcr", "data_valid"])
        self.assertFalse(result.loc["low-bcr", "in_reference_scope"])
        self.assertIn("below_reference_bcr", result.loc["low-bcr", "exclusion_codes"])
        self.assertFalse(result.loc["over-one", "data_valid"])
        self.assertIn("bcr_above_one", result.loc["over-one", "exclusion_codes"])
        self.assertFalse(result.loc["low-coverage", "data_valid"])
        self.assertIn(
            "upstream_quality_failure",
            result.loc["low-coverage", "exclusion_codes"],
        )

    def test_frozen_m1_is_the_eleven_feature_luxembourg_baseline(self):
        self.assertEqual(len(FROZEN_M1_FEATURES), 11)
        self.assertNotIn("volume_density", FROZEN_M1_FEATURES)


if __name__ == "__main__":
    unittest.main()
