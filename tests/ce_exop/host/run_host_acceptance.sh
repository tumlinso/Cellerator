#!/usr/bin/env bash
set -euo pipefail

repo_root="${1:-.}"
cxx="${CXX:-g++}"
scratch="$(mktemp -d "${TMPDIR:-/tmp}/ce-exop-host.XXXXXX")"
trap 'rm -rf -- "${scratch}"' EXIT

cd "${repo_root}"

common=(-std=c++17 -Wall -Wextra -Werror -Iinclude)
strict=(-O2)
sanitized=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)

build_and_run() {
    local mode="$1"
    local name="$2"
    shift 2
    local -a flags=("${common[@]}")
    if [[ "${mode}" == sanitized ]]; then
        flags+=("${sanitized[@]}")
    else
        flags+=("${strict[@]}")
    fi
    "${cxx}" "${flags[@]}" "$@" -o "${scratch}/${name}-${mode}"
    if [[ "${mode}" == sanitized ]]; then
        ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
        UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
            "${scratch}/${name}-${mode}"
    else
        "${scratch}/${name}-${mode}"
    fi
}

operation_sources=(
    src/compute/operation/operation_core_v2/schema.cc
    src/compute/operation/operation_core_v2/relation_algebra.cc
    src/compute/operation/operation_core_v2/composition.cc
    src/compute/operation/operation_core_v2/v1_adapter.cc
)
acquisition_sources=(
    src/execution/geometry_acquisition_v2/schema.cc
    src/execution/geometry_acquisition_v2/assembly.cc
    src/execution/geometry_acquisition_v2/projections.cc
)
boundary_sources=(
    "${acquisition_sources[@]}"
    src/execution/opaque_artifact.cc
    src/geometry/persistence/execution_image_v2.cc
    src/compute/architecture/provider_registry.cc
)

for mode in strict sanitized; do
    build_and_run "${mode}" legacy-baseline \
        tests/ce_exop/host/legacy_baseline_regression_test.cc \
        "${operation_sources[@]}"
    build_and_run "${mode}" scalable-index \
        tests/geometry/ce_exop/scalable_index_validation_test.cc \
        src/geometry/validation/scalable_validation_v2.cc
    build_and_run "${mode}" optimizer-oracle \
        tests/geometry/ce_exop/optimizer_oracle_quality_test.cc \
        src/geometry/optimizer/oracle/exact_oracle.cc
    build_and_run "${mode}" boundary-negative \
        tests/ce_exop/host/boundary_negative_test.cc \
        "${boundary_sources[@]}"
    build_and_run "${mode}" operation-core-v2-integrated \
        src/compute/operation/operation_core_v2/operation_core_v2_test.cc \
        "${operation_sources[@]}"
    build_and_run "${mode}" acquisition-v2-integrated \
        src/execution/geometry_acquisition_v2/geometry_acquisition_v2_test.cc \
        "${acquisition_sources[@]}" \
        src/execution/geometry_acquisition_v2/external_payload.cc
    build_and_run "${mode}" relation-algebra-contract \
        tests/relation_algebra/contract_test.cc
done

tests/ce_exop/host/static_architecture_audit.sh .
git diff --check -- tests/ce_exop/host tests/geometry/ce_exop tests/relation_algebra/v2

echo "CE-EXOP integrated host acceptance: PASS"
echo "compiler=$(${cxx} --version | head -n 1)"
echo "modes=strict,asan+ubsan"
echo "cuda_validation=not_run"
