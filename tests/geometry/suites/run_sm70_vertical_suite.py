#!/usr/bin/env python3
"""Run the source-linked sm_70 first-vertical integration binaries."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


VERTICAL_BINARIES = (
    "geometry_sm70_provider_test",
    "geometry_sm70_value_pack_test",
    "geometry_sm70_n64_test",
    "geometry_sm70_hybrid_test",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", default="build", type=Path)
    arguments = parser.parse_args()
    missing = [
        str(arguments.build / name)
        for name in VERTICAL_BINARIES
        if not (arguments.build / name).is_file()
    ]
    if missing:
        parser.error("missing sm_70 vertical binaries: " + ", ".join(missing))
    for name in VERTICAL_BINARIES:
        binary = arguments.build / name
        print(f"[CE-GEO sm70 vertical] {binary}", flush=True)
        subprocess.run([str(binary)], check=True)
    print(f"CE-GEO sm70 vertical suite passed {len(VERTICAL_BINARIES)} binaries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
