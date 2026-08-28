#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)
build_root=${CELLERATOR_BUILD_DIR:-"$repo_root/build"}
output_dir=${CELLERATOR_CE_LIVE_OUTPUT_DIR:-"$repo_root/build-ce-live-32"}

mkdir -p "$output_dir"
nvcc -std=c++17 -arch=sm_70 \
    -I"$repo_root/include" \
    -I"$build_root/generated" \
    "$repo_root/tests/tensor_core/v100_dense_fragment_candidate_test.cu" \
    "$repo_root/src/compute/candidate/tensor_core/v100_dense_fragment_candidate.cu" \
    "$repo_root/src/compute/candidate/tensor_core/v100_dense_fragment_plan.cc" \
    "$repo_root/src/compute/operation/operation_core.cc" \
    -lcudart \
    -o "$output_dir/v100_dense_fragment_candidate_test"
