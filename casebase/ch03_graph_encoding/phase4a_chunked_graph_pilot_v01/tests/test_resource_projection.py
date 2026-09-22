from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


PHASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PHASE_DIR))

from analyze_phase4a import project_resources


class ResourceProjectionTests(unittest.TestCase):
    def test_projection_uses_each_city_stratum_weight_not_simple_average(self):
        status = pd.DataFrame(
            [
                {
                    "city": "A", "city_count_stratum": 1, "status": "completed",
                    "sample_weight": 10.0, "query_seconds": 1.0,
                    "encoding_seconds": 2.0, "runtime_seconds": 3.0,
                    "output_bytes": 100, "python_peak_memory_mb": 20.0,
                },
                {
                    "city": "B", "city_count_stratum": 4, "status": "completed",
                    "sample_weight": 100.0, "query_seconds": 4.0,
                    "encoding_seconds": 6.0, "runtime_seconds": 10.0,
                    "output_bytes": 1000, "python_peak_memory_mb": 80.0,
                },
            ]
        )

        result = project_resources(status)

        self.assertEqual(result["projected_query_seconds"], 410.0)
        self.assertEqual(result["projected_encoding_seconds"], 620.0)
        self.assertEqual(result["projected_total_runtime_seconds"], 1030.0)
        self.assertEqual(result["projected_graph_bytes"], 101000.0)
        self.assertEqual(result["projected_graph_file_count"], 110.0)
        self.assertEqual(result["peak_python_memory_mb"], 80.0)

    def test_smoke_projection_expands_one_case_per_cell_to_pilot_design(self):
        status = pd.DataFrame(
            [
                {
                    "city": "A", "city_count_stratum": 1, "status": "completed",
                    "sample_weight": 10.0, "query_seconds": 1.0,
                    "encoding_seconds": 2.0, "runtime_seconds": 3.0,
                    "output_bytes": 100, "python_peak_memory_mb": 20.0,
                }
            ]
        )

        result = project_resources(status, weight_multiplier=25)

        self.assertEqual(result["projected_graph_file_count"], 250.0)
        self.assertEqual(result["projected_total_runtime_seconds"], 750.0)


if __name__ == "__main__":
    unittest.main()
