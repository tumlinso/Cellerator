#!/usr/bin/env python3
"""Run the reconciled CE-GEO portable foundation contract binaries."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


FOUNDATION_BINARIES = (
    "ceGeoCapabilityTest",
    "ceGeoProviderRegistryTest",
    "ceGeoProviderContractTest",
    "ceGeoCatalogContractTest",
    "ceGeoProjectionReferenceV2Test",
    "ceGeoBuiltinFragmentTest",
    "ceGeoRelationCatalogTest",
    "ceGeoSemanticCompilerTest",
    "ceGeoCsg1Test",
    "ceGeoCsg1SupportSectionsTest",
    "ceGeoCpe2CapabilityManifestTest",
    "ceGeoPreboundProjectionV2Test",
    "ceGeoGeometryAcquisitionContractTest",
    "ceGeoAcquisitionCostMappingTest",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", default="build", type=Path)
    arguments = parser.parse_args()

    missing = [
        str(arguments.build / name)
        for name in FOUNDATION_BINARIES
        if not (arguments.build / name).is_file()
    ]
    if missing:
        parser.error("missing foundation binaries: " + ", ".join(missing))

    for name in FOUNDATION_BINARIES:
        binary = arguments.build / name
        print(f"[CE-GEO foundation] {binary}", flush=True)
        subprocess.run([str(binary)], check=True)

    print(f"CE-GEO foundation suite passed {len(FOUNDATION_BINARIES)} binaries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
