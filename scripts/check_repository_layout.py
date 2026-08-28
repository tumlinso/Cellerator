#!/usr/bin/env python3
"""Check mechanical repository-remap invariants.

Known baseline violations are explicit debt in the JSON allowlist. New
violations fail immediately; later remap phases delete allowlist entries as
they move declarations and link tests through owning targets.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ALLOWLIST_PATH = ROOT / "cmake" / "repository_layout_allowlist.json"

CANONICAL_SOURCE_ROOTS = (
    ROOT / "include" / "Cellerator",
    ROOT / "src",
    ROOT / "modules",
    ROOT / "components" / "CelleraTorch",
)
IMPLEMENTATION_SCAN_ROOTS = (
    ROOT / "tests",
    ROOT / "bench",
    ROOT / "components" / "CelleraTorch" / "tests",
)
HISTORICAL_CPP_SUFFIXES = {".cpp", ".hpp", ".cxx", ".hxx", ".ixx", ".cppm"}
PUBLIC_HEADER_SUFFIXES = {".h", ".hh", ".cuh", ".inl", ".tcc"}
TRANSLATION_UNIT_SUFFIXES = {".c", ".cc", ".cpp", ".cxx", ".cu"}

INCLUDE_PATTERN = re.compile(r'^\s*#\s*include\s*[<\"]([^>\"]+)[>\"]', re.MULTILINE)
MODULE_IMPORT_PATTERN = re.compile(r"^\s*(?:export\s+)?import\s+", re.MULTILINE)


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def source_files(root: Path):
    if not root.exists():
        return
    for path in root.rglob("*"):
        if path.is_file() and not any(part.startswith("build") for part in path.parts):
            yield path


def load_allowlist() -> dict[str, set[str]]:
    raw = json.loads(ALLOWLIST_PATH.read_text(encoding="utf-8"))
    return {key: set(values) for key, values in raw.items()}


def main() -> int:
    allowlist = load_allowlist()
    errors: list[str] = []
    observed_public_private: set[str] = set()
    observed_implementation_includes: set[str] = set()

    for root in CANONICAL_SOURCE_ROOTS:
        for path in source_files(root):
            rel = relative(path)
            if path.suffix in HISTORICAL_CPP_SUFFIXES:
                errors.append(f"historical C++ extension in canonical tree: {rel}")

            if path.suffix in {".cu", ".cuh"}:
                text = path.read_text(encoding="utf-8", errors="replace")
                if MODULE_IMPORT_PATTERN.search(text):
                    errors.append(f"CUDA source imports a C++ module: {rel}")

            if root == ROOT / "include" / "Cellerator" and path.suffix in PUBLIC_HEADER_SUFFIXES:
                text = path.read_text(encoding="utf-8", errors="replace")
                includes = INCLUDE_PATTERN.findall(text)
                if any("src/" in include for include in includes):
                    observed_public_private.add(rel)
                    if rel not in allowlist["public_headers_including_private_src"]:
                        errors.append(f"public header includes private src path: {rel}")

    for root in IMPLEMENTATION_SCAN_ROOTS:
        for path in source_files(root):
            if path.suffix not in TRANSLATION_UNIT_SUFFIXES:
                continue
            rel = relative(path)
            text = path.read_text(encoding="utf-8", errors="replace")
            includes = INCLUDE_PATTERN.findall(text)
            if any(Path(include).suffix in TRANSLATION_UNIT_SUFFIXES for include in includes):
                observed_implementation_includes.add(rel)
                if rel not in allowlist["translation_units_including_implementation"]:
                    errors.append(f"translation unit includes implementation: {rel}")

    stale_public = allowlist["public_headers_including_private_src"] - observed_public_private
    stale_implementation = (
        allowlist["translation_units_including_implementation"]
        - observed_implementation_includes
    )
    for rel in sorted(stale_public):
        errors.append(f"stale public/private include allowlist entry: {rel}")
    for rel in sorted(stale_implementation):
        errors.append(f"stale implementation include allowlist entry: {rel}")

    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1

    print(
        "repository layout check passed "
        f"({len(observed_public_private)} public/private and "
        f"{len(observed_implementation_includes)} implementation-include debts allowlisted)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
