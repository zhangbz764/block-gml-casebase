from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


PHASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PHASE_DIR))

from build_review_sample import build_review_manifest
from render_review_cases import _case_title, _render_contact_sheets, build_review_html


def review_fixture() -> pd.DataFrame:
    rows = []
    for index in range(120):
        rows.append(
            {
                "city": f"City{index % 8}",
                "block_id": f"B{index:03d}",
                "area_m2": 1000.0 + index * 10,
                "elongation": 0.1 + index / 1000,
                "compactness": 0.2 + index / 1000,
                "lod1_building_count": 10 + index,
                "bcr": 0.01 + index / 1000,
                "hard_flag_count": 0,
                "hard_reasons": "",
                "scope_flag_count": 0,
                "scope_reasons": "",
                "machine_status": "keep_provisional",
                **{f"H{i:02d}": False for i in range(1, 6)},
                **{f"S{i:02d}": False for i in range(1, 7)},
            }
        )
    frame = pd.DataFrame(rows)
    for offset, rule in enumerate(("S01", "S02", "S03", "S04", "S05", "S06")):
        indices = [(offset * 17 + step) % len(frame) for step in range(24)]
        frame.loc[indices, rule] = True
    frame["scope_flag_count"] = frame[[f"S{i:02d}" for i in range(1, 7)]].sum(axis=1)
    frame.loc[frame["scope_flag_count"].gt(0), "machine_status"] = "review_scope"
    frame["scope_reasons"] = frame.apply(
        lambda row: ";".join(f"S{i:02d}" for i in range(1, 7) if row[f"S{i:02d}"]), axis=1
    )
    for offset, rule in enumerate(("H03", "H04", "H05")):
        indices = list(range(108 + offset * 4, 112 + offset * 4))
        frame.loc[indices, rule] = True
        frame.loc[indices, "hard_flag_count"] = 1
        frame.loc[indices, "hard_reasons"] = rule
        frame.loc[indices, "machine_status"] = "exclude_hard"
    return frame


THRESHOLDS = {
    "area_m2_p99": 2100.0,
    "elongation_p01": 0.11,
    "compactness_p01": 0.21,
    "lod1_building_count_p99": 120.0,
    "bcr_p01": 0.02,
}


class ReviewSamplingTests(unittest.TestCase):
    def test_paginated_title_adds_extended_review_fields_without_changing_default_title(self):
        case = pd.DataFrame(
            [
                {
                    "extended_case_no": 7,
                    "city": "Test City",
                    "block_id": "B007",
                    "lod1_building_count": 321,
                    "area_m2": 4567.8,
                    "sample_origin": "city_balance",
                    "hard_reasons": "",
                    "scope_reasons": "building_count_upper_tail",
                }
            ]
        ).itertuples(index=False).__next__()

        self.assertEqual(_case_title(case, paginated=False), "Test City | B007\nbuilding_count_upper_tail")
        paginated = _case_title(case, paginated=True)
        for text in ("#7", "Test City", "B007", "buildings=321", "area=4567.8", "city_balance"):
            self.assertIn(text, paginated)

    def test_contact_sheet_pagination_preserves_default_and_splits_sixty_cases(self):
        rows = []
        block_features = []
        for index in range(60):
            city = f"City{index % 20:02d}"
            block_id = f"B{index:03d}"
            rows.append(
                {
                    "extended_case_no": index + 1,
                    "city": city,
                    "block_id": block_id,
                    "lod1_building_count": 100 + index,
                    "area_m2": 1000.0 + index,
                    "sample_origin": "metric_grid",
                    "hard_reasons": "",
                    "scope_reasons": "building_count_upper_tail",
                    **{f"H{i:02d}": False for i in range(3, 6)},
                    **{f"S{i:02d}": i == 4 for i in range(1, 7)},
                }
            )
            block_features.append(
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]],
                    },
                    "properties": {"city": city, "block_id": block_id},
                }
            )
        manifest = pd.DataFrame(rows)
        blocks = {"type": "FeatureCollection", "features": block_features}
        buildings = {"type": "FeatureCollection", "features": []}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            default_dir = root / "default"
            paged_dir = root / "paged"
            _render_contact_sheets(manifest, blocks, buildings, default_dir)
            _render_contact_sheets(manifest, blocks, buildings, paged_dir, page_size=20)

            self.assertEqual([path.name for path in default_dir.glob("*.png")], ["S04_contact_sheet.png"])
            self.assertEqual(
                sorted(path.name for path in paged_dir.glob("*.png")),
                [
                    "S04_contact_sheet_p01.png",
                    "S04_contact_sheet_p02.png",
                    "S04_contact_sheet_p03.png",
                ],
            )

    def test_review_html_contains_case_identity_and_rule_context(self):
        blocks = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]},
                    "properties": {"city": "Test City", "block_id": "B001", "scope_reasons": "area_upper_tail"},
                }
            ],
        }
        buildings = {"type": "FeatureCollection", "features": []}

        html = build_review_html(blocks, buildings)

        self.assertIn("leaflet", html.lower())
        self.assertIn("Test City", html)
        self.assertIn("B001", html)
        self.assertIn("area_upper_tail", html)

    def test_review_manifest_is_deterministic_unique_and_capped(self):
        frame = review_fixture()
        shuffled = frame.sample(frac=1.0, random_state=19).reset_index(drop=True)

        first = build_review_manifest(frame, THRESHOLDS, max_cases=72)
        second = build_review_manifest(shuffled, THRESHOLDS, max_cases=72)

        pd.testing.assert_frame_equal(first, second)
        self.assertLessEqual(len(first), 72)
        self.assertFalse(first[["city", "block_id"]].duplicated().any())
        self.assertTrue(first["manual_decision"].eq("").all())

    def test_each_rule_respects_limits_and_hard_checks_are_present(self):
        manifest = build_review_manifest(review_fixture(), THRESHOLDS, max_cases=72)

        for rule in ("S01", "S02", "S03", "S04", "S05", "S06"):
            count = manifest["review_sample_roles"].str.contains(rule, regex=False).sum()
            self.assertLessEqual(int(count), 12)
        for rule in ("H03", "H04", "H05"):
            count = manifest["review_sample_roles"].str.contains(rule, regex=False).sum()
            self.assertGreaterEqual(int(count), 1)
            self.assertLessEqual(int(count), 4)


if __name__ == "__main__":
    unittest.main()
