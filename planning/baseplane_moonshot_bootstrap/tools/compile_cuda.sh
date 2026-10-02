#!/usr/bin/env bash
# Compile/disassemble only. This script never runs a kernel or picks a GPU.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="${1:-$ROOT/build-cuda}"
NVCC="${NVCC:-nvcc}"
command -v "$NVCC" >/dev/null || { echo "nvcc unavailable; CUDA seed remains authored/uncompiled" >&2; exit 2; }
ARCHES="$($NVCC --list-gpu-arch)"
grep -q '^compute_70$' <<< "$ARCHES" || { echo "Compiler cannot target sm_70; use the inspected CUDA 12.9 toolchain" >&2; exit 2; }
mkdir -p "$BUILD"
EXTRA=()
if [[ -n "${CUDAHOSTCXX:-}" ]]; then EXTRA+=(-ccbin "$CUDAHOSTCXX"); fi
"$NVCC" -std=c++17 -O2 -lineinfo -arch=sm_70 -Xptxas=-v "${EXTRA[@]}" \
  -I"$ROOT/seed/include" "$ROOT/seed/cuda/smoke.cu" -o "$BUILD/bp_moon_cuda_smoke"
# Use the binary utility beside the chosen nvcc where possible.
NVCC_DIR="$(dirname "$(command -v "$NVCC")")"
if [[ -x "$NVCC_DIR/cuobjdump" ]]; then "$NVCC_DIR/cuobjdump" --dump-sass "$BUILD/bp_moon_cuda_smoke" > "$BUILD/seed.sass"; fi
printf 'Compiled only. Review SASS and use a leased device before running %s --run DEVICE\n' "$BUILD/bp_moon_cuda_smoke"
