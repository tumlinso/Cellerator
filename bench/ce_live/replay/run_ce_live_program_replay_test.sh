#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
configured_build=${CELLERATOR_BUILD_DIR:-"$repo_root/build"}
build_dir=/tmp/cellerator-ce-live-34
cuda_root=/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/cuda/12.9
nvcc="$cuda_root/bin/nvcc"

mkdir -p "$build_dir"
common=(
    -ccbin=/usr/bin/g++-12
    -DCELLSHARD_ENABLE_CUDA=1
    -DCELLSHARD_CUDA_MODE_GENERIC=1
    -DCELLSHARD_CUDA_MODE_NATIVE=0
    -DCELLSHARD_CUDA_MODE_NATIVE_EXTREME=0
    -I"$configured_build/generated"
    -I"$repo_root/include"
    -I"$repo_root"
    -I"/home/tumlinson/CellShard/include"
    -O2 -std=c++17
    -gencode arch=compute_70,code=sm_70
)

if [[ ${1:-} != --run-only ]]; then
    "$nvcc" "${common[@]}" -x cu -c "$repo_root/src/execution/program.cc" \
        -o "$build_dir/program.o"
    "$nvcc" "${common[@]}" -c \
        "$repo_root/tests/persistence/ce_live_program_replay_test.cu" \
        -o "$build_dir/replay_test.o"
    /usr/bin/c++ "$build_dir/program.o" "$build_dir/replay_test.o" \
        -o "$build_dir/ce_live_program_replay_test" \
        -L"$configured_build" \
        -L"$configured_build/src/geometry" \
        -L"$configured_build/CellShard" \
        -L"$cuda_root/targets/x86_64-linux/lib" \
        -L"$cuda_root/../../math_libs/12.9/lib64" \
        -Wl,-rpath,"$cuda_root/targets/x86_64-linux/lib:$cuda_root/../../math_libs/12.9/lib64" \
        -Wl,--start-group \
        -lcellerator_opaque_execution_artifact \
        -lcellerator_projection_activation \
        -lcellerator_preparation_factory \
        -lcellerator_builtin_candidate_catalog \
        -lcellerator_row_masked_n1_candidate \
        -lcellerator_csr_fallback_candidate \
        -lcellerator_feature_major_small_n_candidate \
        -lcellerator_transpose_backward_candidate \
        -lcellerator_transpose_projection \
        -lcellerator_feature_major_projection \
        -lcellerator_operation_core \
        -lcellerator_planner \
        -lcellerator_runtime \
        -lcellpack_execution_image_v2 \
        -lcellpack_persistent_packing_payload \
        -lcellpack_feature_weighted_row_reduction \
        -lcellpack_feature_weighted_row_reduction_cuda \
        -lcellpack_warp_tiles -lcellpack_local_cell_ordering \
        -lcellpack_apply_plan -lcellpack_semantic_geometry -lcellpack \
        -lcellshard_inspect \
        -Wl,--end-group \
        -lcudart -lcublas -lcublasLt -lcusparse -lnvJitLink \
        -lcudadevrt -lcudart_static -lrt -lpthread -ldl -lz -lcrypto -lcurl
fi

if [[ ${1:-} != --build-only ]]; then
    "$build_dir/ce_live_program_replay_test"
fi
