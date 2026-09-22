from __future__ import annotations

import sys
import unittest
from pathlib import Path


PHASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PHASE_DIR))

from batch_db_reader import iter_city_chunks


class BatchReaderTests(unittest.TestCase):
    def test_chunks_never_mix_cities_and_never_exceed_limit(self):
        rows = [
            {"city": "A", "block_id": f"A{i:03d}"} for i in range(205)
        ] + [
            {"city": "B", "block_id": f"B{i:03d}"} for i in range(3)
        ]

        chunks = list(iter_city_chunks(rows, chunk_size=100))

        self.assertEqual([len(item[2]) for item in chunks], [100, 100, 5, 3])
        for city, _, chunk in chunks:
            self.assertEqual({row["city"] for row in chunk}, {city})

    def test_chunking_is_stable_under_input_shuffle(self):
        rows = [
            {"city": city, "block_id": f"{city}{index:03d}"}
            for city in ("B", "A")
            for index in range(103)
        ]
        shuffled = list(reversed(rows))

        self.assertEqual(
            list(iter_city_chunks(rows, 100)),
            list(iter_city_chunks(shuffled, 100)),
        )


if __name__ == "__main__":
    unittest.main()
