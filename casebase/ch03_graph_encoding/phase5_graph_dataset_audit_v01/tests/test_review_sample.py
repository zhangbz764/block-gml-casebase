from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from select_review_sample import select_review_cases
from render_visual_review import build_review_html, render_review_case, write_manual_review
sys.path.insert(0, str(HERE))
from fixtures import normal_record


class ReviewSampleTests(unittest.TestCase):
    def test_two_stable_unique_cases_per_city(self):
        rows = []
        for city_index in range(24):
            city = f"City {city_index:02d}"
            for case in range(3):
                rows.append({"sample_id": f"S{city_index:02d}_{case}", "city": city, "block_id": f"B{case}", "part_id": "part_01",
                             "case_status": "completed", "graph_path": "x", "node_count": case + 1,
                             "connected_components": 1 + (case == 2), "isolated_node_ratio": case / 10,
                             "combined_weak_candidate_ratio": case / 10, "excluded_building_ratio": 0})
        frame = pd.DataFrame(rows)
        first = select_review_cases(frame); second = select_review_cases(frame)
        self.assertEqual(len(first), 48)
        self.assertTrue(first["sample_id"].is_unique)
        self.assertTrue(first.groupby("city").size().eq(2).all())
        self.assertEqual(set(first["selection_source"]), {"city_median", "city_structural_extreme"})
        pd.testing.assert_frame_equal(first, second)

    def test_synthetic_review_render_and_local_html(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metadata = {"sample_id": "S1", "city": "Test City", "block_id": "B1", "node_count": 2,
                        "edge_count": 1, "connected_components": 1, "combined_weak_candidate_count": 0,
                        "excluded_building_count": 0, "selection_source": "city_median", "image_path": "images/S1.png"}
            image = root / "images/S1.png"
            render_review_case(normal_record(), image, metadata)
            manifest = pd.DataFrame([metadata])
            build_review_html(manifest, root / "visual_review.html")
            write_manual_review(manifest, root / "manual_review.csv")
            self.assertGreater(image.stat().st_size, 0)
            self.assertIn("images/S1.png", (root / "visual_review.html").read_text(encoding="utf-8"))
            self.assertEqual(list(pd.read_csv(root / "manual_review.csv").columns), ["sample_id", "visual_status", "issue_type", "comment"])


if __name__ == "__main__": unittest.main()
