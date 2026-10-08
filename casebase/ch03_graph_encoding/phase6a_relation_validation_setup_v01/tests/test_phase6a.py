"""Small behavioral checks; fixtures are not research cases or model results."""

from copy import deepcopy
import gzip
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

PHASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PHASE_DIR))


def pairing_spec():
    return {
        "seed_material": "ch03-p6a-v01|", "max_per_city": 1000,
        "pair_fields": {name: {"transform": "log" if name in ("area_m2", "node_count", "avh_w") else "identity"}
                        for name in ("area_m2", "node_count", "bcr", "far", "compactness", "elongation", "avh_w", "orientation_entropy")},
        "min_active_fields": 6, "nearest_neighbors": 3, "close_max": 0.5,
        "relaxed_max": 1.0, "max_pairs": 30, "max_pairs_per_city": 3,
        "nearest_only_min_pairs": 20,
    }


def index_fixture():
    rows = []
    for city in ("City A", "City B"):
        for i in range(10):
            rows.append({
                "sample_id": f"{city}-{i:02d}", "city": city, "country": "TC",
                "block_id": str(i), "part_id": "part_01", "graph_path": f"graphs/{i}.json.gz",
                "case_status": "completed", "audit_passed": "True",
                "area_m2": 100 + i, "node_count": 1 + i, "bcr": 0.1 + i / 100,
                "far": 1 + i / 100, "compactness": 0.5 + i / 100,
                "elongation": 1 + i / 100, "avh_w": 10 + i / 100,
                "orientation_entropy": i / 100, "lod1_building_count": 1 + i,
                "excluded_building_count": 0,
            })
    return pd.DataFrame(rows)


def graph_fixture():
    core = {name: 0.5 for name in (
        "footprint_compactness", "footprint_rectangularity", "footprint_convexity",
        "orientation_sin_2theta", "orientation_cos_2theta", "local_x_normalized",
        "local_y_normalized", "boundary_distance_normalized",
    )}
    nodes = [
        {**core, "building_id": "B", "height_m": None, "footprint_area_m2": 4,
         "representative_point_x_m": 7, "representative_point_y_m": 2,
         "geometry_wkt_metric": "POLYGON ((6 1,8 1,8 3,6 3,6 1))",
         "missing": {"height_m": True}, "missing_reasons": {"height_m": "source_null"}},
        {**core, "building_id": "A", "height_m": 10, "footprint_area_m2": 4,
         "representative_point_x_m": 2, "representative_point_y_m": 2,
         "geometry_wkt_metric": "POLYGON ((1 1,3 1,3 3,1 3,1 1))",
         "missing": {}, "missing_reasons": {}},
    ]
    values = {name: 0.5 for name in (
        "elongation", "compactness", "bcr", "convexity", "rectangularity", "far", "bcd", "osr",
        "avh_w", "hcv", "footprint_cv", "avg_complexity", "avg_rectangularity", "avg_convexity", "orientation_entropy",
    )}
    values.update({"area_m2": 100, "avh_w": 10, "lod1_building_count": 3})
    edge = {
        "edge_id": "A--B", "source": "B", "target": "A", "footprint_gap_m": 3,
        "footprint_gap_block_norm": 0.3, "centroid_distance_block_norm": 0.5,
        "shared_boundary_ratio_min": 0.1, "shared_boundary_ratio_mean": 0.12,
        "orientation_diff_deg": 30, "orientation_similarity": 0.866,
        "long_side_overlap_ratio": 0.2, "long_side_offset_norm": 0.3,
        "connection_axis_alignment_i": 10, "connection_axis_alignment_j": 70,
        "height_diff_m": None, "log_area_ratio_abs": 0, "log_height_ratio_abs": None,
        "volume_ratio": None, "missing": {"height_diff_m": True, "log_height_ratio_abs": True, "volume_ratio": True},
        "missing_reasons": {"height_diff_m": "source_null", "log_height_ratio_abs": "source_null", "volume_ratio": "source_null"},
        "long_distance_candidate": False, "low_shared_candidate": True, "combined_weak_candidate": False,
    }
    return {
        "encoding_version": "0.3.2", "validation": {"passed": True},
        "sample": {"sample_id": "FIXTURE", "city": "Test City", "block_id": "B0"},
        "graph": {"coordinate_scale_m": 10, "node_count": 2, "excluded_building_count": 1,
                  "block_geometry_wkt_metric": "POLYGON ((0 0,10 0,10 10,0 10,0 0))"},
        "block_attributes": {"values": values, "missing": {}, "missing_reasons": {}},
        "nodes": nodes, "cells": [{"building_id": "A", "geometry_wkt_metric": "POLYGON ((0 0,5 0,5 10,0 10,0 0))"}],
        "cell_failures": [{"building_id": "B", "reason": "degenerate_cell"}], "edges": [edge],
    }


class PairingTests(unittest.TestCase):
    def test_sampling_is_order_invariant_and_parses_false_strings(self):
        from prepare_pairs import sample_development
        frame = index_fixture()
        frame.loc[0, "audit_passed"] = "False"
        frame.loc[1, "case_status"] = "excluded_no_valid_building"
        spec = pairing_spec(); spec["max_per_city"] = 4
        first = sample_development(frame, spec)
        second = sample_development(frame.sample(frac=1, random_state=2), spec)
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(first.groupby("city").size().to_dict(), {"City A": 4, "City B": 4})
        self.assertNotIn("City A-00", first["sample_id"].tolist())
        self.assertNotIn("City A-01", first["sample_id"].tolist())

    def test_pairs_are_same_city_disjoint_and_deterministic(self):
        from prepare_pairs import select_candidate_pairs
        frame = index_fixture(); spec = pairing_spec()
        pairs = select_candidate_pairs(frame, spec)
        self.assertGreater(len(pairs), 0)
        pd.testing.assert_frame_equal(pairs, select_candidate_pairs(frame.iloc[::-1], spec))
        ids = [value for row in pairs.to_dict("records") for value in (row["sample_id_a"], row["sample_id_b"])]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertLessEqual(int(pairs.groupby("city").size().max()), 3)
        lookup = frame.set_index("sample_id")["city"].to_dict()
        self.assertTrue(all(lookup[r.sample_id_a] == lookup[r.sample_id_b] == r.city for r in pairs.itertuples()))

    def test_constant_and_missing_fields_leave_reasons_without_imputation(self):
        from prepare_pairs import select_candidate_pairs
        frame = index_fixture().loc[lambda x: x.city.eq("City A")].copy()
        for name in ("far", "compactness", "elongation"):
            frame[name] = 1
        pairs = select_candidate_pairs(frame, pairing_spec())
        self.assertTrue(pairs.empty)
        self.assertIn("insufficient_active_fields", pairs.attrs["summary"]["cities"]["City A"]["reason"])
        frame = index_fixture(); frame.loc[0, "area_m2"] = np.nan
        pairs = select_candidate_pairs(frame, pairing_spec())
        self.assertNotIn("City A-00", set(pairs.sample_id_a) | set(pairs.sample_id_b))
        self.assertEqual(pairs.attrs["summary"]["cities"]["City A"]["missing_required_fields"], 1)

    def test_nearest_only_is_not_expanded_to_thirty_or_resampled(self):
        from prepare_pairs import select_candidate_pairs
        # Ten independent cities, six possible pairs per city; all distances exceed the grade limits.
        frames = []
        for number in range(10):
            frame = index_fixture().iloc[:6].copy()
            frame["city"] = f"C{number:02d}"
            frame["sample_id"] = [f"C{number:02d}-{i}" for i in range(6)]
            frames.append(frame)
        spec = pairing_spec(); spec["close_max"] = 0; spec["relaxed_max"] = 0
        pairs = select_candidate_pairs(pd.concat(frames, ignore_index=True), spec)
        self.assertEqual(len(pairs), 20)
        self.assertEqual(set(pairs["candidate_level"]), {"nearest_only"})
        self.assertEqual(pairs.attrs["summary"]["selected_by_city"], {f"C{i:02d}": 2 for i in range(10)})


class InputTests(unittest.TestCase):
    def test_block_mapping_shared_nodes_volume_and_missing_masks(self):
        from model_inputs import extract_model_inputs
        result = extract_model_inputs(graph_fixture())
        indicator, sets, graph = result["indicator"], result["building_set"], result["relation_graph"]
        self.assertEqual(indicator["block"]["values"]["avh_w"], 10)
        self.assertEqual(indicator["conditions"]["values"], {"area_m2": 100.0, "node_count": 2.0, "coordinate_scale_m": 10.0})
        self.assertEqual(sets["nodes"], graph["nodes"])
        self.assertEqual(sets["block"], graph["block"])
        self.assertEqual(sets["conditions"], indicator["conditions"])
        self.assertEqual(sets["nodes"]["node_ids"], ["A", "B"])
        self.assertEqual(sets["nodes"]["raw_values"][0]["approx_volume_m3"], 40)
        self.assertEqual(sets["nodes"]["values"][0]["footprint_area_block_norm"], 0.04)
        self.assertEqual(sets["nodes"]["values"][0]["height_scale_norm"], 1)
        self.assertEqual(sets["nodes"]["values"][0]["approx_volume_block_norm"], 0.04)
        self.assertIsNone(sets["nodes"]["values"][1]["height_scale_norm"])
        self.assertTrue(sets["nodes"]["missing_mask"][1]["height_scale_norm"])
        self.assertIsNone(sets["nodes"]["raw_values"][1]["approx_volume_m3"])
        self.assertNotIn("city", indicator["block"]["values"])
        self.assertNotIn("building_id", sets["nodes"]["values"][0])
        self.assertEqual(graph["edge_index"], [[1, 0]])

    def test_bidirectional_edges_exchange_endpoint_attributes_and_masks(self):
        from model_inputs import extract_model_inputs, expand_bidirectional
        graph = extract_model_inputs(graph_fixture())["relation_graph"]
        directed = expand_bidirectional(graph)
        self.assertEqual(directed["edge_index"], [[1, 0], [0, 1]])
        self.assertEqual(directed["edge_values"][1]["connection_axis_alignment_i"], 70)
        self.assertEqual(directed["edge_values"][1]["connection_axis_alignment_j"], 10)
        self.assertTrue(directed["edge_missing_mask"][1]["volume_ratio"])
        graph["edge_missing_mask"][0]["connection_axis_alignment_i"] = True
        graph["edge_missing_mask"][0]["connection_axis_alignment_j"] = False
        directed = expand_bidirectional(graph)
        self.assertFalse(directed["edge_missing_mask"][1]["connection_axis_alignment_i"])
        self.assertTrue(directed["edge_missing_mask"][1]["connection_axis_alignment_j"])

    def test_graph_path_uses_phase4b_root_and_rejects_escape(self):
        from render_pairs import load_graph
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "results/dataset/graphs/fixture.graph.json.gz"
            path.parent.mkdir(parents=True)
            with gzip.open(path, "wt", encoding="utf-8") as handle:
                json.dump(graph_fixture(), handle)
            before = path.read_bytes()
            loaded = load_graph({"graph_path": "results/dataset/graphs/fixture.graph.json.gz"}, root)
            self.assertEqual(loaded["sample"]["sample_id"], "FIXTURE")
            self.assertEqual(path.read_bytes(), before)
            with self.assertRaises(ValueError):
                load_graph({"graph_path": "../outside.json.gz"}, root)
            with self.assertRaises(ValueError):
                load_graph({"graph_path": "results/dataset/graphs/fixture.graph.json.gz", "sample_id": "wrong"}, root)
            record = graph_fixture(); record["validation"]["passed"] = "False"
            with gzip.open(path, "wt", encoding="utf-8") as handle:
                json.dump(record, handle)
            with self.assertRaises(ValueError):
                load_graph({"graph_path": "results/dataset/graphs/fixture.graph.json.gz"}, root)

    def test_field_contract_matches_extracted_input_groups(self):
        from model_inputs import BLOCK_FIELDS, NODE_CORE_FIELDS, NODE_DERIVED_FIELDS, EDGE_FIELDS
        table = pd.read_csv(PHASE_DIR / "model_input_fields_v01.csv")
        self.assertFalse(table["logical_field"].duplicated().any())
        common = set(table.loc[table["indicator"].eq("yes"), "logical_field"])
        self.assertEqual(common, {f"block.{f}" for f in BLOCK_FIELDS} |
                         {"condition.area_m2", "condition.node_count", "condition.coordinate_scale_m"})
        node_fields = set(table.loc[table["building_set"].eq("yes") & table.logical_field.str.startswith("node."), "logical_field"])
        self.assertEqual(node_fields, {f"node.{f}" for f in (*NODE_CORE_FIELDS, *NODE_DERIVED_FIELDS)})
        self.assertEqual(set(table.loc[table.role.eq("core") & table.logical_field.str.startswith("edge."), "logical_field"]),
                         {f"edge.{f}" for f in EDGE_FIELDS})


class RenderingTests(unittest.TestCase):
    def test_same_pair_has_equal_meter_per_pixel_and_height_scale(self):
        from render_pairs import draw_pair_view
        import matplotlib.pyplot as plt
        a = graph_fixture(); b = deepcopy(a)
        b["nodes"][1]["height_m"] = 25
        b["graph"]["block_geometry_wkt_metric"] = "POLYGON ((100 0,130 0,130 10,100 10,100 0))"
        figure, axes, info = draw_pair_view(a, b, mode="plan", show_cells=False)
        figure.canvas.draw()
        self.assertEqual(axes[0].get_aspect(), 1.0)
        self.assertEqual(axes[1].get_aspect(), 1.0)
        spans = [(ax.get_xlim()[1] - ax.get_xlim()[0], ax.get_ylim()[1] - ax.get_ylim()[0]) for ax in axes]
        np.testing.assert_allclose(spans[0], spans[1])
        scale = [spans[i][0] / axes[i].get_window_extent().width for i in range(2)]
        self.assertAlmostEqual(scale[0], scale[1])
        self.assertEqual(info["height_min_m"], 10)
        self.assertEqual(info["height_max_m"], 25)
        self.assertEqual(info["missing_height_color"], "#bdbdbd")
        plt.close(figure)

    def test_no_edges_and_missing_cells_preserve_nodes_and_null_statistics(self):
        from model_inputs import extract_model_inputs
        from render_pairs import relation_summary, draw_pair_view
        import matplotlib.pyplot as plt
        record = graph_fixture()
        record["nodes"] = record["nodes"][1:]
        record["edges"] = []; record["cells"] = []
        record["graph"]["node_count"] = 1
        rows = relation_summary(record)
        self.assertTrue(all(row["valid_values"] == 0 and row["median"] is None for row in rows))
        self.assertEqual(extract_model_inputs(record)["building_set"]["nodes"]["node_ids"], ["A"])
        fig, axes, _ = draw_pair_view(record, record, mode="weak", show_cells=True)
        self.assertGreater(len(axes[0].patches), 0)
        plt.close(fig)

    def test_example_edges_are_extreme_deduplicated_and_ties_use_ids(self):
        from render_pairs import select_example_edges
        record = graph_fixture()
        record["edges"] = [dict(record["edges"][0], edge_id=f"E{i}", footprint_gap_m=i,
                                height_diff_m=30 if i in (3,4) else 9-i, missing={}) for i in range(8)]
        selected = select_example_edges(record)
        self.assertEqual([edge["edge_id"] for edge in selected], ["E0", "E1", "E3", "E4", "E6", "E7"])


class RunnerTests(unittest.TestCase):
    def test_excluded_entries_need_no_graph_path_or_country_but_completed_entries_do(self):
        from run_phase6a import validate_source_counts
        from prepare_pairs import sample_development
        frame = index_fixture()
        excluded = frame.iloc[:2].copy()
        excluded["sample_id"] = ["EXCLUDED-1", "EXCLUDED-2"]
        excluded["case_status"] = "excluded_no_valid_building"
        excluded["audit_passed"] = "False"
        excluded["graph_path"] = None; excluded["country"] = None
        frame = pd.concat([frame, excluded], ignore_index=True)
        expected = {"entries": 22, "completed": 20, "excluded_no_valid_building": 2, "cities": 2}
        self.assertEqual(validate_source_counts(frame, expected), expected)
        self.assertEqual(len(sample_development(frame, pairing_spec())), 20)
        frame.loc[0, "graph_path"] = None
        with self.assertRaises(ValueError):
            validate_source_counts(frame, expected)

    def test_frozen_index_hash_and_accepted_counts_are_enforced(self):
        from run_phase6a import read_frozen_index, sha256
        spec = json.loads((PHASE_DIR / "pairing_spec_v01.json").read_text(encoding="utf-8"))
        spec.update(expected_entries=20, expected_completed=20, expected_excluded=0, expected_cities=2)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root / "results/dataset").mkdir(parents=True)
            index = root / "results/dataset/graph_dataset_index.csv.gz"
            index_fixture().to_csv(index, index=False, compression="gzip")
            freeze = root / "results/freeze.json"
            freeze.write_text(json.dumps({"output_sha256": {"dataset/graph_dataset_index.csv.gz": sha256(index)},
                                         "evidence_state": "MANUAL_REVIEW_PENDING"}), encoding="utf-8")
            frame, _ = read_frozen_index(root, spec)
            self.assertEqual(len(frame), 20)
            spec["expected_completed"] = 19
            with self.assertRaises(ValueError):
                read_frozen_index(root, spec)
            freeze.write_text(json.dumps({"output_sha256": {"dataset/graph_dataset_index.csv.gz": "wrong"}}), encoding="utf-8")
            with self.assertRaises(ValueError):
                read_frozen_index(root, spec)

    def test_nonempty_run_directory_and_unsafe_id_are_rejected(self):
        from run_phase6a import reserve_run_directory
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = reserve_run_directory(root, "candidate_v01")
            (path / "pair_review.csv").write_text("researcher decision", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                reserve_run_directory(root, "candidate_v01")
            with self.assertRaises(ValueError):
                reserve_run_directory(root, "../escape")

    def test_manual_pipeline_outputs_blank_judgments_and_does_not_modify_sources(self):
        import run_phase6a
        from prepare_pairs import sample_development, select_candidate_pairs
        spec = json.loads((PHASE_DIR / "pairing_spec_v01.json").read_text(encoding="utf-8"))
        spec["max_pairs"] = 1
        frame = index_fixture()
        pair = select_candidate_pairs(sample_development(frame, spec), spec).iloc[0].to_dict()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); phase = root / "phase6a"; phase.mkdir()
            source = root / "phase4b"; before = {}
            (phase / "pairing_spec_v01.json").write_text(json.dumps(spec), encoding="utf-8")
            for side in ("a", "b"):
                record = graph_fixture()
                record["sample"].update({field: pair[f"{field}_{side}"] for field in ("sample_id", "city", "block_id")})
                path = source / pair[f"graph_path_{side}"]; path.parent.mkdir(parents=True, exist_ok=True)
                with gzip.open(path, "wt", encoding="utf-8") as handle:
                    json.dump(record, handle)
                before[path] = path.read_bytes()
            with patch.object(run_phase6a, "PHASE_DIR", phase), patch.object(run_phase6a, "PHASE4B_DIR", source), \
                 patch.object(run_phase6a, "read_frozen_index", return_value=(frame, {"index_sha256": "fixture-only"})):
                page = run_phase6a.run("candidates", "candidate_test", 1)
                with self.assertRaises(FileExistsError):
                    run_phase6a.run("candidates", "candidate_test", 1)
            run_dir = page.parent
            review = pd.read_csv(run_dir / "pair_review.csv")
            self.assertTrue(review.drop(columns="pair_id").isna().all().all())
            checks = json.loads((run_dir / "verification_summary.json").read_text(encoding="utf-8"))
            self.assertTrue(checks["program_checks_passed"])
            self.assertEqual(checks["checked_graphs"], 2)
            self.assertFalse(checks["model_executed"])
            self.assertEqual(len(list((run_dir / "images").glob("*.png"))), 5)
            document = page.read_text(encoding="utf-8")
            self.assertIn("images/", document)
            self.assertIn(".join('\\r\\n')", document)  # Valid escaped JS, not literal newlines in a string.
            self.assertNotIn("http://", document)
            for path, data in before.items():
                self.assertEqual(path.read_bytes(), data)


if __name__ == "__main__":
    unittest.main()
