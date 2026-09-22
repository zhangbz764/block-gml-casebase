from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


PHASE_DIR = Path(__file__).resolve().parents[1]
SOURCE = PHASE_DIR.parent / "phase3_5b_final_sample_freeze_v01" / "results" / "dataset" / "graph_encoding_manifest.csv.gz"
sys.path.insert(0, str(PHASE_DIR))

from build_full_manifests import assign_parts, load_group_spec


EXPECTED = {
    "part_01": 28892, "part_02": 28891, "part_03": 29362,
    "part_04": 23073, "part_05": 16610, "part_06": 12620,
    "part_07": 12878, "part_08": 12085, "part_09": 12180,
    "part_10": 12002, "part_11": 12014, "part_12": 12447,
}


class FullManifestTests(unittest.TestCase):
    def test_real_frozen_manifest_is_partitioned_exactly_once(self):
        source = pd.read_csv(SOURCE, encoding="utf-8-sig", low_memory=False)
        result = assign_parts(source, load_group_spec())

        self.assertEqual(len(result), 213054)
        self.assertFalse(result[["city", "block_id"]].duplicated().any())
        self.assertEqual(result.groupby("part_id").size().to_dict(), EXPECTED)
        city_parts = result.groupby("city")["part_id"].nunique()
        self.assertEqual(int(city_parts.drop("Tokyo").max()), 1)
        self.assertEqual(int(city_parts["Tokyo"]), 2)


if __name__ == "__main__":
    unittest.main()
