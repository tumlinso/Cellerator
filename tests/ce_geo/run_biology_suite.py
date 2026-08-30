#!/usr/bin/env python3
"""Run the integrated relation-algebra and sm_70 biology operation suite."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


BIOLOGY_BINARIES = (
    "ceGeoRelationCatalogIntegrationTest",
    "ceGeoRelationContractTest",
    "ceGeoEdgeMapOrGateTest",
    "ceGeoRelationInterfaceTest",
    "ceGeoRelationCompatibilityTest",
    "ceGeoRelationBundleTest",
    "ceGeoSegmentNormalizeTest",
    "ceGeoSegmentReduceTest",
    "ceGeoSm70AdvancedOpsTest",
    "ceGeoSm70EdgeValueGradientTest",
    "ceGeoSm70SegmentBackwardTest",
    "ceGeoSm70CoverNormalizeTest",
    "ceGeoSm70TransposeApplyTest",
    "ceGeoBiologicalRelationsExample",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", default="build", type=Path)
    arguments = parser.parse_args()
    missing = [
        str(arguments.build / name)
        for name in BIOLOGY_BINARIES
        if not (arguments.build / name).is_file()
    ]
    if missing:
        parser.error("missing biology binaries: " + ", ".join(missing))
    for name in BIOLOGY_BINARIES:
        binary = arguments.build / name
        print(f"[CE-GEO biology] {binary}", flush=True)
        subprocess.run([str(binary)], check=True)
    print(f"CE-GEO biology suite passed {len(BIOLOGY_BINARIES)} binaries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
