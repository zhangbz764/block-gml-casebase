"""Create deterministic SHA-256 inventories for phase 2.5 and phase 3 outputs."""

from __future__ import annotations

import csv
import hashlib
from datetime import datetime, timezone
from pathlib import Path


PHASE3_DIR = Path(__file__).resolve().parent
TARGETS = [
    PHASE3_DIR.parent / "phase2_5_tessellation_relations_v02" / "results",
    PHASE3_DIR / "results",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_inventory(directory: Path) -> None:
    output = directory / "output_inventory.csv"
    files = sorted(
        path
        for path in directory.rglob("*")
        if path.is_file()
        and path != output
        and "__pycache__" not in path.parts
        and path.suffix != ".pyc"
    )
    rows = []
    for path in files:
        stat = path.stat()
        rows.append(
            {
                "relative_path": path.relative_to(directory).as_posix(),
                "type": path.suffix.lstrip(".") or "file",
                "size_bytes": stat.st_size,
                "sha256": sha256_file(path),
                "modified_at": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
            }
        )
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"{directory.name}: {len(rows)} files")


if __name__ == "__main__":
    for target in TARGETS:
        build_inventory(target)
