from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


PHASE_DIR = Path(__file__).resolve().parents[1]
TESTS_DIR = Path(__file__).resolve().parent
SOURCE_LEDGER = (
    PHASE_DIR.parent
    / "phase3_5a_block_screening_v01"
    / "results"
    / "bulk"
    / "full_v01"
    / "block_screening_ledger.csv.gz"
)
sys.path.insert(0, str(PHASE_DIR))
sys.path.insert(0, str(TESTS_DIR))

from freeze_final_sample import freeze_run
from verify_final_sample import verify_run

from test_final_screening import base_row


class OutputPipelineTests(unittest.TestCase):
    def test_small_run_is_written_and_independently_verified(self):
        rows = [base_row(f"B{i:03d}", lod1_building_count=1) for i in range(198)]
        rows.extend(
            [
                base_row("TOP1", lod1_building_count=369, S04=True),
                base_row("TOP2", lod1_building_count=369),
                base_row("S01", S01=True),
            ]
        )
        source = pd.DataFrame(rows)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            source_path = temp / "source.csv.gz"
            source.to_csv(source_path, index=False, encoding="utf-8-sig", compression="gzip")

            result = freeze_run(
                source_path=source_path,
                results_dir=temp / "results",
                source_run_id="fixture_v01",
                run_id="test_v01",
                enforce_expected=False,
            )
            verification = verify_run(
                results_dir=temp / "results",
                run_id="test_v01",
                source_run_id="fixture_v01",
                enforce_expected=False,
            )

            self.assertEqual(result["building_count_p995_threshold"], 369.0)
            self.assertTrue(verification["passed"])
            self.assertEqual(verification["final_excluded"], 3)
            self.assertEqual(verification["final_eligible"], 198)
            self.assertTrue(
                (temp / "results" / "dataset" / "graph_encoding_manifest.csv.gz").exists()
            )

    @unittest.skipUnless(SOURCE_LEDGER.exists(), "Phase 3.5A full ledger is unavailable")
    def test_real_input_matches_frozen_phase_counts(self):
        source = pd.read_csv(
            SOURCE_LEDGER,
            encoding="utf-8-sig",
            float_precision="round_trip",
            low_memory=False,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            result = freeze_run(
                source_path=SOURCE_LEDGER,
                results_dir=Path(temp_dir) / "results",
                source_run_id="full_v01",
                run_id="final_v01",
                enforce_expected=True,
            )

        self.assertEqual(len(source), 219419)
        self.assertEqual(result["building_count_p995_threshold"], 369.0)
        self.assertEqual(result["final_excluded"], 6365)
        self.assertEqual(result["final_eligible"], 213054)
        self.assertEqual(result["final_eligible_building_nodes"], 7158522)


if __name__ == "__main__":
    unittest.main()
