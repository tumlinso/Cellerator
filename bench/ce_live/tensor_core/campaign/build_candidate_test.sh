#!/usr/bin/env bash
set -euo pipefail

mkdir -p build-ce-live-32
nvcc -std=c++17 -arch=sm_70 \
    -Iinclude \
    -Icomponents/CellPack/include \
    tests/tensor_core/v100_dense_fragment_candidate_test.cu \
    src/compute/math/tensor_core/v100_dense_fragment_candidate.cu \
    src/compute/math/tensor_core/v100_dense_fragment_plan.cc \
    src/compute/math/operation_core/operation_core.cc \
    -lcudart \
    -o build-ce-live-32/v100_dense_fragment_candidate_test
