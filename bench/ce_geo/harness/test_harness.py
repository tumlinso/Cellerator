#!/usr/bin/env python3
"""CPU-only CE-GEO-110 gate entrypoint."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

from validate_evidence import validate_command_manifest


def main() -> int:
    harness = Path(__file__).resolve().parent
    schema = harness.parent / "evidence" / "schema"
    for path in sorted(schema.glob("*.json")):
        with path.open("r", encoding="utf-8") as stream:
            json.load(stream)

    with (schema / "command_manifest.example.json").open(
            "r", encoding="utf-8") as stream:
        result = validate_command_manifest(json.load(stream))
    if result.get("performance_run") is not False:
        raise AssertionError("schema validation attempted a performance run")

    suite = unittest.defaultTestLoader.discover(
        str(harness), pattern="test_*.py")
    outcome = unittest.TextTestRunner(verbosity=2).run(suite)
    if not outcome.wasSuccessful():
        return 1
    print("CE-GEO-110-HARNESS passed; hardware_query=false performance_run=false")
    return 0


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    raise SystemExit(main())
