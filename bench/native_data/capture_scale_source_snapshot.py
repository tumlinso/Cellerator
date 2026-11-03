#!/usr/bin/env python3
"""Capture the source archive and small harness through Autobench unchanged."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from autobench.registry import parse_registration
from autobench.snapshot import capture


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REGISTRATION = Path(__file__).with_name("scale-correctness.toml")
SOURCE_ARCHIVE = (
    ".autobench/data/cellerator-native-data-gated-product-scale/source-closure-v3.tar"
)


def main() -> int:
    benchmark = parse_registration(REGISTRATION, PROJECT_ROOT)
    build_snapshot = replace(
        benchmark,
        include=("bench/native_data/**",),
        data=(SOURCE_ARCHIVE,),
    )
    snapshot = capture(build_snapshot, PROJECT_ROOT / ".autobench/state")
    print(f"fingerprint={snapshot.fingerprint}")
    print(f"snapshot={snapshot.directory}")
    print(f"files={len(snapshot.manifest)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
