#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="${1:-$ROOT/build-host}"
cmake -S "$ROOT/seed" -B "$BUILD" -DCMAKE_BUILD_TYPE=Debug
cmake --build "$BUILD" -j 2
ctest --test-dir "$BUILD" --output-on-failure
"$BUILD/bp_moon_demo"
