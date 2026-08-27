#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
build_dir=/tmp/cellerator-ce-live-31
cuda_root=/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/cuda/12.9
nvcc="$cuda_root/bin/nvcc"
fixture="$repo_root/bench/ce_live/fixture/local/pbmc3k-r512-s7.npz"
fixture_bin="$build_dir/pbmc3k-r512-s7.bin"

mkdir -p "$build_dir"
python "$repo_root/bench/ce_live/forward/prepare_pbmc3k_fixture.py" \
    --fixture "$fixture" \
    --manifest "$repo_root/data/manifests/ce_live/pbmc3k_quantitative_v1.json" \
    --output "$fixture_bin"

common=(
    -ccbin=/usr/bin/g++-12
    -I"$repo_root/build-dissolution-smoke/generated"
    -I"$repo_root/include"
    -I"$repo_root/components/CellPack/include"
    -I"$repo_root"
    -O2
    -std=c++17
    -gencode arch=compute_70,code=sm_70
)

if [[ ${1:-} != --run-only ]]; then
    "$nvcc" "${common[@]}" -x cu -c "$repo_root/src/execution/program.cc" \
        -o "$build_dir/program.o"
    "$nvcc" "${common[@]}" -c \
        "$repo_root/tests/live/quantitative_forward_test.cu" \
        -o "$build_dir/quantitative_forward_test.o"
    /usr/bin/c++ "$build_dir/program.o" \
        "$build_dir/quantitative_forward_test.o" \
        -o "$build_dir/quantitative_forward_test" \
        -L"$repo_root/build-dissolution-smoke" \
        -L"$repo_root/build-dissolution-smoke/components/CellPack" \
        -L"$cuda_root/targets/x86_64-linux/lib" \
        -L"$cuda_root/../../math_libs/12.9/lib64" \
        -Wl,-rpath,"$cuda_root/targets/x86_64-linux/lib:$cuda_root/../../math_libs/12.9/lib64" \
        -Wl,--start-group \
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
        -lcellerator_identity_registry \
        -lcellerator_live_runtime_fixture \
        -lcellpack_persistent_packing_payload \
        -lcellpack_feature_weighted_row_reduction \
        -lcellpack_feature_weighted_row_reduction_cuda \
        -lcellpack_warp_tiles \
        -lcellpack_local_cell_ordering \
        -lcellpack_apply_plan \
        -lcellpack \
        -Wl,--end-group \
        -lcudart -lcublas -lcublasLt -lcusparse -lnvJitLink \
        -lcudadevrt -lcudart_static -lrt -lpthread -ldl
fi

if [[ ${1:-} != --build-only ]]; then
    "$build_dir/quantitative_forward_test" "$fixture_bin" "${2:-5}"
fi
