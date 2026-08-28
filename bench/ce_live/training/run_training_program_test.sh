#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
configured_build=${CELLERATOR_BUILD_DIR:-"$repo_root/build"}
build_dir=/tmp/cellerator-ce-live-33
cuda_root=/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/cuda/12.9
nvcc="$cuda_root/bin/nvcc"

mkdir -p "$build_dir"

common=(
    -ccbin=/usr/bin/g++-12
    -I"$configured_build/generated"
    -I"$repo_root/include"
    -O2
    -std=c++17
    -gencode arch=compute_70,code=sm_70
)

"$nvcc" "${common[@]}" -c "$repo_root/src/execution/training_program.cu" \
    -o "$build_dir/training_program.o"
"$nvcc" "${common[@]}" -c "$repo_root/tests/execution/training_program_test.cu" \
    -o "$build_dir/training_program_test.o"

/usr/bin/c++ "$build_dir/training_program.o" \
    "$build_dir/training_program_test.o" \
    -o "$build_dir/training_program_test" \
    -L"$configured_build" \
    -L"$configured_build/src/geometry" \
    -L"$cuda_root/targets/x86_64-linux/lib" \
    -L"$cuda_root/../../math_libs/12.9/lib64" \
    -Wl,-rpath,"$cuda_root/targets/x86_64-linux/lib:$cuda_root/../../math_libs/12.9/lib64" \
    -lcellerator_native_training_slice \
    -lcellerator_feature_major_projection \
    -lcellerator_transpose_projection \
    -lcellpack_persistent_packing_payload \
    -lcellpack_feature_weighted_row_reduction \
    -lcellpack_warp_tiles \
    -lcellpack_local_cell_ordering \
    -lcellpack_apply_plan \
    -lcellpack \
    -lcellerator_runtime \
    -lcudart -lcublas -lcublasLt -lcusparse -lnvJitLink \
    -lcudadevrt -lcudart_static -lrt -lpthread -ldl

if [[ ${1:-} != --build-only ]]; then
    "$build_dir/training_program_test"
fi
