from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
from build_graph_index import _latest_status, build_part_index, phase5_code_bundle_sha256
from fixtures import META, normal_record


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PartIndexTests(unittest.TestCase):
    def test_phase5_code_bundle_is_stable_and_detects_source_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); first = root / "b.py"; second = root / "a.py"
            first.write_text("b = 1\n", encoding="utf-8"); second.write_text("a = 1\n", encoding="utf-8")
            before = phase5_code_bundle_sha256([first, second])
            self.assertEqual(before, phase5_code_bundle_sha256([second, first]))
            second.write_text("a = 2\n", encoding="utf-8")
            self.assertNotEqual(before, phase5_code_bundle_sha256([first, second]))

    def test_latest_attempt_wins(self):
        frame = pd.DataFrame([{"sample_id": "S", "attempt": "1", "status": "failed"}, {"sample_id": "S", "attempt": "2", "status": "completed"}])
        self.assertEqual(_latest_status(frame).iloc[0]["status"], "completed")

    def test_build_and_resume_part(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); input_root = root / "phase4b"; output = root / "phase5"; phase = HERE.parent
            manifest_path = input_root / "results/dataset/parts/part_01_manifest.csv.gz"
            manifest_path.parent.mkdir(parents=True)
            manifest = pd.DataFrame([
                {"sample_id": "S1", "city": "Test City", "country": "TC", "block_id": "B1", "part_id": "part_01", "lod1_building_count": "2"},
                {"sample_id": "S2", "city": "Test City", "country": "TC", "block_id": "B2", "part_id": "part_01", "lod1_building_count": "1"},
            ])
            manifest.to_csv(manifest_path, index=False, encoding="utf-8-sig", compression="gzip")
            meta = dict(META); meta.update(source_manifest_sha256="source", code_bundle_sha256="code", part_manifest_sha256=digest(manifest_path))
            graph_path = input_root / "graphs/S1.graph.json.gz"; graph_path.parent.mkdir()
            record = normal_record(); record["phase4b"] = dict(meta)
            with gzip.open(graph_path, "wt", encoding="utf-8") as handle: json.dump(record, handle)
            run = input_root / "results/runs/full_v032/part_01"; run.mkdir(parents=True)
            status = pd.DataFrame([
                {"sample_id": "S1", "city": "Test City", "block_id": "B1", "part_id": "part_01", "attempt": "1", "status": "failed"},
                {"sample_id": "S1", "city": "Test City", "block_id": "B1", "part_id": "part_01", "attempt": "2", "status": "completed", "graph_path": "graphs/S1.graph.json.gz", "output_bytes": graph_path.stat().st_size, "planned_building_count": "2", "excluded_building_count": "0"},
                {"sample_id": "S2", "city": "Test City", "block_id": "B2", "part_id": "part_01", "attempt": "1", "status": "excluded_no_valid_building", "planned_building_count": "1", "excluded_building_count": "1"},
            ])
            status.to_csv(run / "encoding_status.csv", index=False, encoding="utf-8-sig")
            (run / "preflight.json").write_text(json.dumps({"phase4b": meta}), encoding="utf-8")
            spec = {"phase4b_code_bundle_sha256": "code", "phase5_code_bundle_sha256": "phase5-code-v1", "phase35b_manifest_sha256": "source", "phase4b_full_manifest_sha256": "full", "formal_output_name": "full_v01", "_spec_path": str(phase / "audit_spec_v01.json")}
            result = build_part_index("part_01", input_root, output, spec)
            issue_text = (output / "audit/full_v01/parts/part_01_issues.csv").read_text(encoding="utf-8-sig")
            self.assertTrue(result["passed"], issue_text); self.assertEqual(result["index_rows"], 2)
            index = pd.read_csv(output / "dataset/index_parts/part_01_graph_index.csv.gz")
            self.assertEqual(set(index["case_status"]), {"completed", "excluded_no_valid_building"})
            self.assertTrue(build_part_index("part_01", input_root, output, spec)["skipped"])
            changed_code = {**spec, "phase5_code_bundle_sha256": "phase5-code-v2"}
            with self.assertRaisesRegex(RuntimeError, "incompatible inputs"):
                build_part_index("part_01", input_root, output, changed_code)
            status.loc[len(status)] = {"sample_id": "X", "attempt": "1", "status": "failed"}
            status.to_csv(run / "encoding_status.csv", index=False, encoding="utf-8-sig")
            with self.assertRaisesRegex(RuntimeError, "incompatible inputs"):
                build_part_index("part_01", input_root, output, spec)


if __name__ == "__main__":
    unittest.main()
