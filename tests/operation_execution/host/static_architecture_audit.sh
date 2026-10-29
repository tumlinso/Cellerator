#!/usr/bin/env bash
set -euo pipefail

repo_root="${1:-.}"

abi_paths=(
    include/Cellerator/compute/operation/operation_core_v2
    include/Cellerator/compute/operation/operation_core_v2.hh
    include/Cellerator/compute/operation/candidate_catalog_v2.hh
    include/Cellerator/execution/geometry_acquisition_v2
    include/Cellerator/execution/geometry_acquisition_v2.hh
    include/Cellerator/execution/opaque_artifact.hh
    include/Cellerator/geometry/index/scalable_views_v2.hh
    include/Cellerator/geometry/validation/scalable_validation_v2.hh
    include/Cellerator/geometry/optimizer/oracle
    include/Cellerator/geometry/optimizer/portfolio_v1.hh
)

production_paths=(
    "${abi_paths[@]}"
    src/compute/operation/operation_core_v2
    src/execution/geometry_acquisition_v2
    src/execution/opaque_artifact.cc
    src/geometry/validation/scalable_validation_v2.cc
    src/geometry/optimizer/oracle
)

cd "${repo_root}"

fail_if_match() {
    local description="$1"
    local expression="$2"
    shift 2
    if rg -n --glob '*.{h,hh,hpp,c,cc,cpp,cu,cuh}' "${expression}" "$@"; then
        echo "architecture audit failed: ${description}" >&2
        exit 1
    fi
}

# Stable ABI is pointer-light caller-owned POD. Allocator-owning STL containers
# or implicit ownership are a contract regression even when used only cold.
fail_if_match "allocator-owning STL in stable ABI" \
    'std::(vector|map|unordered_map|unordered_set|set|list|deque|string|shared_ptr|unique_ptr)' \
    "${abi_paths[@]}"
fail_if_match "implicit allocation in stable ABI" \
    '\b(new|delete|malloc|calloc|realloc|free)\b[[:space:]]*[(\[]' \
    "${abi_paths[@]}"

# The execution portfolio must remain below framework/model policy and must not
# acquire storage or transport ownership from CellShard.
fail_if_match "framework or model ownership in execution portfolio" \
    '#[[:space:]]*include[[:space:]]*[<\"](torch|ATen|Cellerator/(model|models|framework))' \
    "${production_paths[@]}"
fail_if_match "CellShard ownership leak" \
    '#[[:space:]]*include[[:space:]]*[<\"]CellShard/' \
    "${production_paths[@]}"

# Architecture dispatch belongs to providers and compiled projections, not the
# stable semantic ABI. Global compiler policy must likewise stay out of it.
fail_if_match "architecture macro in stable semantic ABI" \
    '(__CUDA_ARCH__|__CUDACC_VER|sm_70|compute_70)' \
    "${abi_paths[@]}"
fail_if_match "global fast-math policy" \
    '(^|[[:space:]])(--use_fast_math|-ffast-math)([[:space:]]|$)' \
    "${production_paths[@]}"

echo "CE-EXOP static architecture audit: PASS"
echo "audited_abi_paths=${#abi_paths[@]}"
echo "audited_production_paths=${#production_paths[@]}"
