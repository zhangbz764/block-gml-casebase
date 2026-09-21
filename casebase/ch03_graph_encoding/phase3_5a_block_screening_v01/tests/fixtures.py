from __future__ import annotations

from typing import Any

import pandas as pd


def valid_row(**overrides: Any) -> dict[str, Any]:
    row = {
        "city": "Test City",
        "block_id": "B001",
        "area_m2": 1000.0,
        "elongation": 0.5,
        "compactness": 0.5,
        "lod1_building_count": 2,
        "bcr": 0.4,
        "convexity": 0.8,
        "rectangularity": 0.8,
        "far": 1.0,
        "bcd": 20.0,
        "osr": 0.6,
        "avh": 10.0,
        "avh_w": 10.0,
        "hsd": 1.0,
        "hcv": 0.1,
        "h_max": 15.0,
        "h_range": 5.0,
        "total_volume": 1000.0,
        "avg_volume": 500.0,
        "avg_footprint": 100.0,
        "footprint_cv": 0.1,
        "avg_complexity": 1.2,
        "avg_rectangularity": 0.8,
        "avg_convexity": 0.9,
        "avg_orientation": 45.0,
        "orientation_entropy": 1.0,
        "avg_nn_distance": 5.0,
        "geom_is_null": False,
        "geom_is_empty": False,
        "geom_is_valid": True,
        "geom_area_positive": True,
    }
    row.update(overrides)
    return row


def mixed_fixture() -> pd.DataFrame:
    rows = [
        valid_row(block_id="KEEP", area_m2=1000.0),
        valid_row(block_id="HARD", bcd=0.0),
    ]
    rows.extend(
        valid_row(block_id=f"N{index:03d}", area_m2=100.0 + index)
        for index in range(100)
    )
    rows.append(valid_row(block_id="EXTREME", area_m2=1_000_000.0))
    return pd.DataFrame(rows)
