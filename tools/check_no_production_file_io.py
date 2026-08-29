#!/usr/bin/env python3
"""Reject ordinary storage ownership in shipped Cellerator library sources."""

from __future__ import annotations

import pathlib
import re
import sys


SOURCE_SUFFIXES = {".h", ".hh", ".hpp", ".cuh", ".c", ".cc", ".cpp", ".cu"}
RULES = (
    ("C++ file streams", re.compile(r"<fstream>|std::(?:i|o|f)stream\b")),
    ("C++ filesystem", re.compile(r"<filesystem>|std::filesystem\b")),
    ("C/POSIX file API", re.compile(r"\b(?:fopen|open|read|write)\s*\(")),
    ("HDF5 API", re.compile(r"(?:#\s*include\s*[<\"]hdf5|\bH5[A-Z][A-Za-z0-9_]*\s*\()")),
    ("CellShard csh5 I/O", re.compile(r"CellShard/io/csh5|\.csh5\b")),
    ("CellShard storage export", re.compile(r"CellShard/export/")),
)


def main() -> int:
    root = pathlib.Path(__file__).resolve().parents[1]
    violations: list[str] = []
    for source_root in (root / "include", root / "src"):
        for path in sorted(source_root.rglob("*")):
            if not path.is_file() or path.suffix not in SOURCE_SUFFIXES:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            for line_number, line in enumerate(text.splitlines(), 1):
                for label, pattern in RULES:
                    if pattern.search(line):
                        violations.append(
                            f"{path.relative_to(root)}:{line_number}: {label}: {line.strip()}"
                        )
    if violations:
        print("Cellerator production file-I/O boundary violations:", file=sys.stderr)
        print("\n".join(violations), file=sys.stderr)
        return 1
    print("Cellerator production file-I/O boundary: clean")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
