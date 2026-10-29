#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
binary="${TMPDIR:-/tmp}/ce-live-24-quantitative-relation-test"

cd "$repo_root"
nvcc -std=c++17 -arch=sm_70 -lineinfo -Iinclude -I. \
    tests/live/quantitative_relation_test.cu \
    tests/fixtures/model_execution/quantitative_relation_fixture.cc \
    src/structure/identity_registry.cc \
    -o "$binary"
"$binary"
/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/cuda/12.9/compute-sanitizer/compute-sanitizer \
    --tool memcheck --error-exitcode 99 "$binary"
