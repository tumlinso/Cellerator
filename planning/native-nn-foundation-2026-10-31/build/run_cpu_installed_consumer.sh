#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
build_dir="${1:-/tmp/cellerator-native-nn-cmake}"
validation_root="$(mktemp -d "${TMPDIR:-/tmp}/cellerator-native-nn-consumer.XXXXXX")"
install_prefix="${validation_root}/install"
consumer_build="${validation_root}/consumer"

python_bin="${PYTHON:-python3}"
torch_dir="${Torch_DIR:-}"
torch_lib_dir="${TORCH_LIB_DIR:-}"
if [[ -z "${torch_dir}" ]]; then
    torch_dir="$("${python_bin}" -c 'import torch; print(torch.utils.cmake_prefix_path + "/Torch")')"
fi
if [[ -z "${torch_lib_dir}" ]]; then
    torch_lib_dir="$("${python_bin}" -c 'import os, torch; print(os.path.join(os.path.dirname(torch.__file__), "lib"))')"
fi

cuda_toolkit_root="${CUDAToolkit_ROOT:-${CUDA_HOME:-}}"
if [[ -z "${cuda_toolkit_root}" && -d /opt/nvidia/hpc_sdk/Linux_x86_64/26.1/cuda/12.9 ]]; then
    cuda_toolkit_root="/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/cuda/12.9"
fi

cmake --install "${build_dir}" --prefix "${install_prefix}"

prefix_path="${install_prefix}"
if [[ -n "${CONDA_PREFIX:-}" ]]; then
    prefix_path="${prefix_path};${CONDA_PREFIX}"
fi

cmake_args=(
    -S "${repo_root}/planning/native-nn-foundation-2026-10-31/build/consumer"
    -B "${consumer_build}"
    -DCMAKE_PREFIX_PATH="${prefix_path}"
    -DTorch_DIR="${torch_dir}"
    -DCMAKE_CXX_COMPILER="${CXX:-/usr/bin/g++-12}"
)
if [[ -n "${cuda_toolkit_root}" ]]; then
    cmake_args+=(
        -DCUDA_TOOLKIT_ROOT_DIR="${cuda_toolkit_root}"
        -DCUDA_NVCC_EXECUTABLE="${cuda_toolkit_root}/bin/nvcc"
        -DCUDAToolkit_ROOT="${cuda_toolkit_root}"
        -DCMAKE_CUDA_COMPILER="${cuda_toolkit_root}/bin/nvcc"
    )
fi
cmake "${cmake_args[@]}"
cmake --build "${consumer_build}" --target nn_installed_consumer -j2
LD_LIBRARY_PATH="${torch_lib_dir}${CONDA_PREFIX:+:${CONDA_PREFIX}/lib}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
    "${consumer_build}/nn_installed_consumer"

printf 'Installed consumer prefix: %s\n' "${install_prefix}"
printf 'Installed consumer build: %s\n' "${consumer_build}"
