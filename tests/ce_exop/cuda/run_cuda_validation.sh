#!/usr/bin/env bash
set -euo pipefail

repo_root="${1:-.}"
build_dir="${2:-${repo_root}/build-ce-exop-valid-cuda}"
mode="${3:---build-only}"
suite="${4:-candidate-matrix}"

candidate_matrix_tests=(
    "ceGeoSm70WidthsTest|tests/tensor_core/sm70/relation_apply_widths_test.cu|src/compute/architecture/providers/nvidia/sm70/relation_apply_widths.cu"
    "ceGeoSm70N64Test|tests/tensor_core/sm70/relation_apply_n64_test.cu|src/compute/architecture/providers/nvidia/sm70/relation_apply_n64.cu"
    "ceGeoSm70ValuePackTest|tests/tensor_core/sm70/value_pack_test.cu|src/compute/architecture/providers/nvidia/sm70/value_pack.cu"
    "ceGeoSm70HybridTest|tests/tensor_core/sm70/relation_apply_hybrid_test.cu|src/compute/architecture/providers/nvidia/sm70/relation_apply_hybrid.cu src/compute/architecture/providers/nvidia/sm70/value_pack.cu src/compute/architecture/providers/nvidia/sm70/relation_apply_n64.cu src/compute/architecture/providers/nvidia/sm70/residual.cu"
)

dynamic_value_tests=(
    "ceGeoEdgeMapOrGateTest|tests/relation_algebra/edge_map_or_gate_test.cu|src/compute/candidate/edge_map_or_gate.cu"
    "ceGeoSm70ValuePackTest|tests/tensor_core/sm70/value_pack_test.cu|src/compute/architecture/providers/nvidia/sm70/value_pack.cu"
)

transpose_gradient_tests=(
    "ceGeoSm70TransposeApplyTest|tests/tensor_core/sm70/transpose_relation_apply_test.cu|"
    "ceGeoSm70ContractOnSupportTest|tests/tensor_core/sm70/contract_on_support_test.cu|"
    "ceGeoSm70EdgeValueGradientTest|tests/tensor_core/sm70/edge_value_gradient_test.cu|"
)

compile_tests() {
    mkdir -p "${build_dir}"
    local record target test_source source_list source
    for record in "$@"; do
        IFS='|' read -r target test_source source_list <<< "${record}"
        # source_list is a controlled repository-local list from this script.
        read -r -a sources <<< "${source_list}"
        absolute_sources=()
        for source in "${sources[@]}"; do
            absolute_sources+=("${repo_root}/${source}")
        done
        nvcc -std=c++17 -arch=sm_70 -lineinfo \
            -Xcompiler=-Wall,-Wextra,-Werror \
            -I"${repo_root}/include" -I"${repo_root}" \
            "${repo_root}/${test_source}" \
            "${absolute_sources[@]}" \
            -o "${build_dir}/${target}"
    done
}

execute_targets() {
    if [[ -z "${PROJECT_CONTROL_GPU_LEASE_ID:-}" ]]; then
        echo "GPU execution requires PROJECT_CONTROL_GPU_LEASE_ID" >&2
        return 2
    fi
    local target
    for target in "$@"; do
        "${build_dir}/${target}"
    done
}

case "${suite}" in
    candidate-matrix)
        selected_tests=("${candidate_matrix_tests[@]}")
        ;;
    dynamic-values)
        selected_tests=("${dynamic_value_tests[@]}")
        ;;
    transpose-gradients)
        selected_tests=("${transpose_gradient_tests[@]}")
        ;;
    *)
        echo "unknown CUDA validation suite: ${suite}" >&2
        exit 2
        ;;
esac

compile_tests "${selected_tests[@]}"

case "${mode}" in
    --build-only)
        ;;
    --execute)
        targets=()
        for record in "${selected_tests[@]}"; do
            targets+=("${record%%|*}")
        done
        execute_targets "${targets[@]}"
        ;;
    *)
        echo "usage: $0 [repo-root] [build-dir] [--build-only|--execute] [candidate-matrix|dynamic-values|transpose-gradients]" >&2
        exit 2
        ;;
esac

echo "CE-EXOP CUDA ${suite} ${mode}: PASS"
echo "cuda_architectures=70"
