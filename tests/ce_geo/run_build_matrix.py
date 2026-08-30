#!/usr/bin/env python3
"""Configure and build the native and explicit compatibility CE-GEO matrix."""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path


def run(command: list[str], cwd: Path) -> None:
    print("[CE-GEO matrix] " + " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", default=1, type=int)
    arguments = parser.parse_args()
    if arguments.jobs < 1:
        parser.error("--jobs must be positive")
    source = Path(__file__).resolve().parents[2]
    configurations = (("native", "OFF"), ("compat", "ON"))
    with tempfile.TemporaryDirectory(prefix="ce-geo-build-matrix-") as root:
        for name, torch_mode in configurations:
            build = Path(root) / name
            run([
                "cmake", "-S", str(source), "-B", str(build),
                "-DCMAKE_BUILD_TYPE=Release",
                "-DCMAKE_CUDA_ARCHITECTURES=70",
                f"-DCELLERATOR_ENABLE_TORCH_MODELS={torch_mode}",
            ], source)
            run([
                "cmake", "--build", str(build),
                "--target", "ceGeoSm70VerticalTests", "ceGeoBiologyTests",
                "-j", str(arguments.jobs),
            ], source)
    print("CE-GEO native/compatibility build matrix passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
