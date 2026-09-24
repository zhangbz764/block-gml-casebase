"""Capture the read-only Phase 4B all-parts verification as the Phase 5 gate."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

from registry import load_audit_spec


PHASE_DIR = Path(__file__).resolve().parent
CH03_DIR = PHASE_DIR.parent
PHASE4B_DIR = CH03_DIR / "phase4b_full_graph_batches_v032"
DEFAULT_OUTPUT = PHASE_DIR / "results" / "audit" / "full_v01" / "phase4b_all_parts_verification.json"


def validate_phase4b_gate(summary: dict, spec: dict) -> list[str]:
    issues: list[str] = []
    missing = list(summary.get("missing_parts", []))
    if missing:
        issues.append(f"Phase 4B parts are missing: {', '.join(missing)}")
    problems = summary.get("problems", {})
    if problems:
        issues.append(f"Phase 4B reports unresolved problems: {sorted(problems)}")
    if summary.get("passed") is not True:
        issues.append("Phase 4B all-parts verification has not passed")
    expected = int(spec["expected_entries"])
    accounted = int(summary.get("accounted_entries", -1))
    if accounted != expected:
        issues.append(f"Phase 4B accounted entries are {accounted}, expected {expected}")
    if int(summary.get("expected_entries", expected)) != expected:
        issues.append("Phase 4B verification uses a different expected population")
    completed = int(summary.get("completed_graphs", 0))
    excluded = int(summary.get("excluded_no_valid_building", 0))
    if completed + excluded != expected:
        issues.append(f"Phase 4B terminal statuses sum to {completed + excluded}, expected {expected}")
    return issues


def _phase4b_verify_all_parts() -> dict:
    module_path = PHASE4B_DIR / "verify_phase4b.py"
    module_spec = importlib.util.spec_from_file_location("phase4b_frozen_verify", module_path)
    if module_spec is None or module_spec.loader is None:
        raise RuntimeError(f"Cannot load frozen Phase 4B verifier: {module_path}")
    module = importlib.util.module_from_spec(module_spec)
    sys.path.insert(0, str(PHASE4B_DIR))
    try:
        module_spec.loader.exec_module(module)
        return module.verify_all_parts()
    finally:
        if sys.path and sys.path[0] == str(PHASE4B_DIR):
            sys.path.pop(0)


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def capture_phase4b_verification(output_path: Path) -> dict:
    summary = _phase4b_verify_all_parts()
    _atomic_json(output_path, summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", action="store_true", required=True)
    args = parser.parse_args()
    if not args.capture:
        return 2
    spec = load_audit_spec(PHASE_DIR / "audit_spec_v01.json")
    summary = capture_phase4b_verification(DEFAULT_OUTPUT)
    issues = validate_phase4b_gate(summary, spec)
    if issues:
        print(json.dumps({"passed": False, "issues": issues}, ensure_ascii=False, indent=2))
        return 1
    print(json.dumps({"passed": True, "output": str(DEFAULT_OUTPUT)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
