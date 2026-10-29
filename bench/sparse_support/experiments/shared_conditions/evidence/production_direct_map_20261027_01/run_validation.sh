#!/bin/sh
set -eu

root=/home/tumlinson/Cellerator
evidence="$root/bench/sparse_support/experiments/shared_conditions/evidence/production_direct_map_20261027_01"
binary="$evidence/cellerator_sparse_support_cuda_worksets.snapshot"
sanitizer=/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/compilers/bin/compute-sanitizer

cd "$root"
sha256sum -c "$evidence/sha256.txt"

printf '%s\n' '=== direct-map CUDA correctness ==='
"$binary"

printf '%s\n' '=== Compute Sanitizer memcheck ==='
"$sanitizer" --tool memcheck --error-exitcode 99 "$binary"

printf '%s\n' '=== Compute Sanitizer racecheck ==='
"$sanitizer" --tool racecheck --error-exitcode 99 "$binary"

printf '%s\n' '=== Compute Sanitizer synccheck ==='
"$sanitizer" --tool synccheck --error-exitcode 99 "$binary"

printf '%s\n' '=== validation complete ==='
