#!/usr/bin/env python3
"""Reject changes to the three fixed CPE2 v2 record sizes."""

import pathlib
import re
import sys


def main() -> int:
    if len(sys.argv) != 4:
        print("usage: check_cpe2_sizes.py <header> <section> <projection>", file=sys.stderr)
        return 2

    expected = tuple(int(value) for value in sys.argv[1:])
    root = pathlib.Path(__file__).resolve().parents[3]
    header = root / "include/Cellerator/geometry/persistence/execution_image_v2.hh"
    text = header.read_text(encoding="utf-8")
    records = (
        "execution_image_v2_header",
        "execution_section_entry_v1",
        "execution_projection_entry_v1",
    )
    actual = []
    for record in records:
        match = re.search(
            rf"static_assert\(sizeof\({record}\) == (\d+)u,", text)
        if match is None:
            print(f"missing fixed-size assertion for {record}", file=sys.stderr)
            return 1
        actual.append(int(match.group(1)))

    if tuple(actual) != expected:
        print(f"CPE2 size mismatch: expected={expected} actual={tuple(actual)}", file=sys.stderr)
        return 1
    print(f"CPE2 fixed sizes passed: header={actual[0]} section={actual[1]} projection={actual[2]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
