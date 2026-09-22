from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


PHASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PHASE_DIR))

from build_pilot_manifest import build_pilot_manifest, build_smoke_manifest


class PilotSamplingTests(unittest.TestCase):
    def setUp(self):
        self.manifest = pd.DataFrame(
            [
                {
                    "city": f"City {city:02d}",
                    "block_id": f"B{city:02d}_{index:03d}",
                    "lod1_building_count": index + 1,
                    "area_m2": 1000.0 + index,
                    "bcr": 0.2,
                }
                for city in range(24)
                for index in range(100)
            ]
        )

    def test_frozen_design_is_unique_weighted_and_has_96_case_smoke(self):
        pilot = build_pilot_manifest(self.manifest, per_stratum=25, seed=20260922)
        smoke = build_smoke_manifest(pilot)

        self.assertEqual(len(pilot), 2400)
        self.assertFalse(pilot[["city", "block_id"]].duplicated().any())
        self.assertEqual(pilot.groupby(["city", "city_count_stratum"]).size().nunique(), 1)
        self.assertEqual(len(smoke), 96)
        self.assertEqual(len(smoke[["city", "city_count_stratum"]].drop_duplicates()), 96)
        self.assertTrue((pilot["sample_weight"] == 1.0).all())

    def test_sampling_is_unchanged_when_input_rows_are_shuffled(self):
        first = build_pilot_manifest(self.manifest, per_stratum=25, seed=20260922)
        shuffled = self.manifest.sample(frac=1, random_state=7).reset_index(drop=True)
        second = build_pilot_manifest(shuffled, per_stratum=25, seed=20260922)

        pd.testing.assert_frame_equal(first, second)


if __name__ == "__main__":
    unittest.main()
