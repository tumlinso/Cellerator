#!/usr/bin/env bash
set -euo pipefail

repo_root=$(git rev-parse --show-toplevel)
build_dir=$(mktemp -d)
trap 'rm -rf "${build_dir}"' EXIT

compiler=${CXX:-c++}
common=(-std=c++17 -Wall -Wextra -Werror -pedantic)
source_file="${repo_root}/tests/profiling/ce_exop/profiler_fixture_validation.cc"
fixture_file="${repo_root}/bench/ce_exop/profiler_fixture_matrix.tsv"
candidate_file="${repo_root}/bench/ce_exop/profiler_candidate_matrix.tsv"

"${compiler}" "${common[@]}" -O2 "${source_file}" -o "${build_dir}/validate"
"${build_dir}/validate" "${fixture_file}" "${candidate_file}"

"${compiler}" "${common[@]}" -O1 -g -fno-omit-frame-pointer \
    -fsanitize=address,undefined "${source_file}" -o "${build_dir}/validate_sanitized"
ASAN_OPTIONS=detect_leaks=1 UBSAN_OPTIONS=halt_on_error=1 \
    "${build_dir}/validate_sanitized" "${fixture_file}" "${candidate_file}"

python3 "${repo_root}/tests/profiling/ce_exop/validate_manifests.py" \
    "${repo_root}/bench/ce_exop"

"${repo_root}/tests/profiling/ce_exop/run_compile_smoke.sh"

python3 "${repo_root}/tests/profiling/ce_exop/validate_generic_export.py" \
    "${repo_root}/bench/ce_exop/generic_partition_export_v1.json"

python3 "${repo_root}/tests/profiling/ce_exop/validate_deferred_requirements.py" \
    "${repo_root}/bench/ce_exop/deferred_profiling_requirements_v1.json"
