#!/usr/bin/env bash
set -euo pipefail

repo_root=$(git rev-parse --show-toplevel)
build_dir=$(mktemp -d)
trap 'find "${build_dir}" -type f -delete; rmdir "${build_dir}"' EXIT

cuda_root=${CUDA_ROOT:-/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/cuda/12.9}
nvcc="${cuda_root}/bin/nvcc"
cuobjdump="${cuda_root}/bin/cuobjdump"
source_file="${repo_root}/tests/profiling/ce_exop/profiler_compile_smoke.cu"

"${nvcc}" -std=c++17 -arch=sm_70 -lineinfo -dc "${source_file}" \
    -o "${build_dir}/markers_off.o"
"${nvcc}" -std=c++17 -arch=sm_70 -lineinfo -dc \
    -DCELLERATOR_ENABLE_PROFILING_MARKERS=1 "${source_file}" \
    -o "${build_dir}/markers_on.o"

"${cuobjdump}" --dump-elf "${build_dir}/markers_off.o" > "${build_dir}/off.elf"
"${cuobjdump}" --dump-elf "${build_dir}/markers_on.o" > "${build_dir}/on.elf"

grep -q '\.nv_debug_line_sass' "${build_dir}/off.elf"
grep -q 'ce_exop_relation_hybrid_mma' "${build_dir}/off.elf"
grep -q 'ce_exop_segment_softmax_max' "${build_dir}/off.elf"
if grep -q 'relation_marker\|segment_marker' "${build_dir}/off.elf"; then
    echo "disabled marker leaked into object" >&2
    exit 1
fi
grep -q 'relation_marker' "${build_dir}/on.elf"
grep -q 'segment_marker' "${build_dir}/on.elf"

echo "compile_only=true"
echo "cuda_runtime_executed=false"
echo "arch=sm_70"
"${nvcc}" --version | tail -n 1
