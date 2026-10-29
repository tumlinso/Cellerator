#!/usr/bin/env python3
"""Run the integrated relation-algebra and sm_70 biology operation suite."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


BIOLOGY_BINARIES = (
    "relation_catalog_integration_test",
    "relation_contract_test",
    "edge_map_or_gate_test",
    "relation_interface_test",
    "relation_compatibility_test",
    "relation_bundle_test",
    "segment_normalize_test",
    "segment_reduce_test",
    "geometry_sm70_advanced_ops_test",
    "geometry_sm70_edge_value_gradient_test",
    "geometry_sm70_segment_backward_test",
    "geometry_sm70_cover_normalize_test",
    "geometry_sm70_transpose_apply_test",
    "biological_relations_example",
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
