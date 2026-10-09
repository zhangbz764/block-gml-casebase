"""Small checks for local adaptation, not a rerun of the historical encoders.

Written before implementation. Not executed in the code-only preparation turn.
"""

from copy import deepcopy
import csv
import gzip
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

PHASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PHASE))


def spec():
    return json.loads((PHASE / "probe_layouts_v01.json").read_text(encoding="utf-8"))


class GeometryTests(unittest.TestCase):
    def test_seven_layouts_preserve_footprints_and_named_center_operations(self):
        from control_probes import make_probe_layouts
        layouts = {item["id"]: item for item in make_probe_layouts(spec())}
        self.assertEqual(len(layouts), 7)
        for layout in layouts.values():
            self.assertEqual(len(layout["buildings"]), 8)
            for index, building in enumerate(layout["buildings"]):
                geometry = building["geometry"]
                self.assertTrue(layout["block_geometry"].contains(geometry))
                self.assertAlmostEqual(geometry.area, 96)
                x0, y0, x1, y1 = geometry.bounds
                self.assertAlmostEqual(x1-x0, 12); self.assertAlmostEqual(y1-y0, 8)
                self.assertTrue(all(not geometry.intersects(other["geometry"])
                                    for other in layout["buildings"][index+1:]))
        expected = {"BASE": (30, 30), "GAP_080": (36, 32), "GAP_120": (24, 28),
                    "OFFSET_08": (26, 30), "OFFSET_16": (22, 30)}
        for name, center in expected.items():
            node = layouts[name]["buildings"][0]
            self.assertEqual((node["geometry"].centroid.x, node["geometry"].centroid.y), center)
        self.assertEqual(layouts["OFFSET_16"]["buildings"][4]["geometry"].centroid.x, 38)

    def test_height_variants_keep_the_same_geometry_and_height_multiset(self):
        from control_probes import make_probe_layouts
        layouts = {item["id"]: item for item in make_probe_layouts(spec())}
        for name, high_ids in (("HEIGHT_ADJ", {"B01", "B02"}), ("HEIGHT_SEP", {"B01", "B08"})):
            buildings = layouts[name]["buildings"]
            self.assertEqual(sorted(b["height"] for b in buildings), [8]*6 + [24]*2)
            self.assertEqual(sum(b["height"] for b in buildings)/8, 12)
            self.assertEqual({b["building_id"] for b in buildings if b["height"] == 24}, high_ids)
            for b, base in zip(buildings, layouts["BASE"]["buildings"], strict=True):
                self.assertTrue(b["geometry"].equals(base["geometry"]))


class EncodingTests(unittest.TestCase):
    def test_local_adapter_encodes_real_relations_without_database_provenance(self):
        from control_probes import make_probe_layouts, encode_probe, summarize_probe
        import psycopg2
        with patch.object(psycopg2, "connect", side_effect=AssertionError("DB not allowed")):
            layouts = make_probe_layouts(spec())
            a, b = encode_probe(layouts[0]), encode_probe(layouts[1])
        for record in (a, b):
            self.assertEqual(record["encoding_version"], "0.3.2")
            self.assertTrue(record["validation"]["passed"])
            self.assertEqual(record["graph"]["excluded_building_count"], 0)
            self.assertEqual(record["graph"]["metric_srid"], 0)
            self.assertEqual(len(record["nodes"]), 8)
            self.assertEqual(record["source_kind"], "synthetic_local")
            self.assertNotIn("lod1.", json.dumps(record["graph"]["source_index"]))
            for node in record["nodes"]:
                self.assertIsNone(node["source_index"]["schema"])
            self.assertIsNone(record["block_attributes"]["values"]["far"])
            self.assertTrue(record["block_attributes"]["missing"]["far"])
            self.assertTrue(record["block_attributes"]["missing_reasons"]["far"])
            ids = {node["building_id"] for node in record["nodes"]}
            for edge in record["edges"]:
                self.assertIn(edge["source"], ids); self.assertIn(edge["target"], ids)
        # Hand-derived same-row footprint gaps: 20-12=8 m versus 16-12=4 m.
        for record, expected_gap in ((a, 8), (b, 4)):
            edge = next(e for e in record["edges"] if {e["source"], e["target"]} == {"B01", "B02"})
            self.assertAlmostEqual(edge["footprint_gap_m"], expected_gap)
            self.assertAlmostEqual(summarize_probe(record)["avh_w_m"], 12)
            self.assertEqual(summarize_probe(record)["footprint_area_total_m2"], 768)

    def test_missing_summaries_and_changed_adjacency_are_not_zero_filled(self):
        from control_probes import summarize_probe, edge_comparisons
        empty = {"graph": {}, "nodes": [], "edges": [], "sample": {"sample_id": "EMPTY"}}
        summary = summarize_probe(empty)
        self.assertIsNone(summary["footprint_gap_m_median"])
        self.assertEqual(summary["footprint_gap_m_valid_count"], 0)
        edge = {"source": "B01", "target": "B05", "footprint_gap_m": 12}
        rows = edge_comparisons({"BASE": dict(empty, edges=[edge]), "OFFSET_08": empty}, "offset")
        absent = next(row for row in rows if row["layout_id"] == "OFFSET_08" and row["endpoint_a"] == "B01")
        self.assertFalse(absent["present"])
        self.assertEqual(absent["adjacency_note"], "邻接改变")
        self.assertIsNone(absent["footprint_gap_m"])


class PresentationTests(unittest.TestCase):
    def test_synthetic_panels_share_extent_scale_and_height_normalization(self):
        from control_probes import make_probe_layouts, encode_probe
        from render_control_probes import draw_plan_panels, draw_height_axons
        import matplotlib.pyplot as plt
        layouts = make_probe_layouts(spec())
        records = [encode_probe(layouts[i]) for i in (0, 1, 2)]
        figure, axes, checks = draw_plan_panels(records, synthetic=True)
        self.assertTrue(all(ax.get_aspect() == 1 for ax in axes))
        self.assertTrue(all(tuple(ax.get_xlim()) == (-5, 125) and tuple(ax.get_ylim()) == (-5, 85) for ax in axes))
        self.assertAlmostEqual(checks["meters_per_pixel"][0], checks["meters_per_pixel"][1])
        self.assertEqual(checks["height_range_m"], [8, 24])
        plt.close(figure)
        figure = draw_height_axons([encode_probe(layouts[5]), encode_probe(layouts[6])])
        self.assertEqual(len(figure.axes), 3)  # Two axon views and the shared height legend.
        plt.close(figure)


class SourceTests(unittest.TestCase):
    def test_only_manifest_cases_are_read_with_phase4b_path_safety(self):
        from control_probes import select_real_cases
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); graph_root = root / "phase4b"; graph_root.mkdir()
            graph = {"encoding_version": "0.3.2", "validation": {"passed": True},
                     "sample": {"sample_id": "KNOWN", "city": "Test", "block_id": "01"},
                     "nodes": [], "edges": [], "graph": {}}
            path = graph_root / "one.json.gz"
            with gzip.open(path, "wt", encoding="utf-8") as handle:
                json.dump(graph, handle)
            manifest = root / "pair_manifest.csv"
            fields = ["city", *[f"{f}_{s}" for s in ("a", "b") for f in ("sample_id", "block_id", "graph_path")]]
            with manifest.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
                writer.writerow({"city": "Test", **{f"{f}_{s}": v for s in ("a", "b") for f,v in
                                (("sample_id", "KNOWN"), ("block_id", "01"), ("graph_path", "one.json.gz"))}})
            before = path.read_bytes()
            cards, sources = select_real_cases(manifest, graph_root)
            self.assertEqual(len(sources), 1)
            self.assertTrue(all(card["record"] is None for card in cards.values()))
            self.assertEqual(path.read_bytes(), before)
            content = manifest.read_text(encoding="utf-8").replace("one.json.gz", "../outside.json.gz")
            manifest.write_text(content, encoding="utf-8")
            with self.assertRaises(ValueError):
                select_real_cases(manifest, graph_root)

    def test_existing_outputs_are_not_overwritten(self):
        from run_phase6b_preparation import reserve_output_dirs
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            probes, _ = reserve_output_dirs(root, "preview_v01")
            (probes / "probe_response.csv").write_text("existing", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                reserve_output_dirs(root, "preview_v01")
            with self.assertRaises(ValueError):
                reserve_output_dirs(root, "../escape")


if __name__ == "__main__":
    unittest.main()
