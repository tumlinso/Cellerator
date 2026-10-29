#!/usr/bin/env python3
"""Run the reconciled CE-GEO portable foundation contract binaries."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


FOUNDATION_BINARIES = (
    "geometry_capability_test",
    "geometry_provider_registry_test",
    "geometry_provider_contract_test",
    "geometry_catalog_contract_test",
    "geometry_projection_reference_v2_test",
    "geometry_builtin_fragment_test",
    "relation_catalog_test",
    "geometry_semantic_compiler_test",
    "geometry_csg1_test",
    "geometry_csg1_support_sections_test",
    "geometry_cpe2_capability_manifest_test",
    "geometry_prebound_projection_v2_test",
    "geometry_acquisition_contract_test",
    "geometry_acquisition_cost_mapping_test",
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
