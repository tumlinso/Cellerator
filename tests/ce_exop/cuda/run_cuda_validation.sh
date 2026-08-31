#!/usr/bin/env bash
set -euo pipefail

repo_root="${1:-.}"
build_dir="${2:-${repo_root}/build-ce-exop-valid-cuda}"
mode="${3:---build-only}"

candidate_matrix_tests=(
    "ceGeoSm70WidthsTest|tests/tensor_core/sm70/relation_apply_widths_test.cu|src/compute/architecture/providers/nvidia/sm70/relation_apply_widths.cu"
    "ceGeoSm70N64Test|tests/tensor_core/sm70/relation_apply_n64_test.cu|src/compute/architecture/providers/nvidia/sm70/relation_apply_n64.cu"
    "ceGeoSm70ValuePackTest|tests/tensor_core/sm70/value_pack_test.cu|src/compute/architecture/providers/nvidia/sm70/value_pack.cu"
    "ceGeoSm70HybridTest|tests/tensor_core/sm70/relation_apply_hybrid_test.cu|src/compute/architecture/providers/nvidia/sm70/relation_apply_hybrid.cu src/compute/architecture/providers/nvidia/sm70/value_pack.cu src/compute/architecture/providers/nvidia/sm70/relation_apply_n64.cu src/compute/architecture/providers/nvidia/sm70/residual.cu"
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

compile_tests "${candidate_matrix_tests[@]}"

case "${mode}" in
    --build-only)
        ;;
    --execute)
        targets=()
        for record in "${candidate_matrix_tests[@]}"; do
            targets+=("${record%%|*}")
        done
        execute_targets "${targets[@]}"
        ;;
    *)
        echo "usage: $0 [repo-root] [build-dir] [--build-only|--execute]" >&2
        exit 2
        ;;
esac

echo "CE-EXOP CUDA candidate matrix ${mode}: PASS"
echo "cuda_architectures=70"
