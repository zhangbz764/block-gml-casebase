"""Write the frozen Phase 4B critical-code manifest before formal runs start."""

from __future__ import annotations

import json
from pathlib import Path

from code_integrity import CH03_DIR, CRITICAL_CODE_FILES, compute_code_manifest


PHASE_DIR = Path(__file__).resolve().parent
OUTPUT = PHASE_DIR / "results" / "audit" / "prep" / "code_hashes.json"
RUN_ROOT = PHASE_DIR / "results" / "runs" / "full_v032"


def main() -> int:
    started = sorted(path.name for path in RUN_ROOT.glob("part_*") if path.is_dir()) if RUN_ROOT.exists() else []
    if started:
        raise RuntimeError(f"Refusing to change frozen code hashes after formal runs started: {started}")
    manifest = compute_code_manifest(CRITICAL_CODE_FILES, CH03_DIR)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_name(OUTPUT.name + ".tmp")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(OUTPUT)
    print(json.dumps({"code_bundle_sha256": manifest["code_bundle_sha256"], "file_count": len(manifest["files"])}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
