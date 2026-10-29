"""Freeze and verify the code bundle used by every formal Phase 4B part."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT=Path(__file__).resolve().parents[1]
CRITICAL_CODE_FILES=tuple(sorted(
    [*ROOT.glob('*.py'),*(ROOT/'core').glob('*.py'),*(ROOT/'pipeline').glob('*.py'),
     *(ROOT/'pipeline').glob('*.sql'),*(ROOT/'specs').glob('*.json'),*(ROOT/'specs').glob('*.tsv')],
    key=lambda p:p.relative_to(ROOT).as_posix()))

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compute_code_manifest(paths: list[Path] | tuple[Path, ...], root: Path) -> dict:
    root = root.resolve()
    files: dict[str, str] = {}
    for path in paths:
        resolved = path.resolve()
        try:
            relative = resolved.relative_to(root).as_posix()
        except ValueError as error:
            raise ValueError(f"Critical code file is outside hash root: {resolved}") from error
        files[relative] = sha256_file(resolved)
    payload = "".join(f"{name}\0{files[name]}\n" for name in sorted(files)).encode("utf-8")
    return {
        "hash_algorithm": "sha256",
        "root": "casebase/graph_encoding",
        "files": dict(sorted(files.items())),
        "code_bundle_sha256": hashlib.sha256(payload).hexdigest(),
    }


def verify_code_manifest(paths: list[Path] | tuple[Path, ...], root: Path, frozen: dict) -> dict:
    actual = compute_code_manifest(paths, root)
    if actual.get("files") != frozen.get("files") or actual["code_bundle_sha256"] != frozen.get("code_bundle_sha256"):
        raise RuntimeError("Critical Phase 4B code differs from the frozen code manifest")
    return actual


def load_and_verify_frozen_code(path: Path) -> dict:
    frozen = json.loads(path.read_text(encoding="utf-8"))
    verify_code_manifest(CRITICAL_CODE_FILES, ROOT, frozen)
    return frozen
