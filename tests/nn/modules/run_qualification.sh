#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "usage: $0 <cellerator-install-prefix> <work-dir> [module-install-prefix]" >&2
    exit 2
fi

cellerator_prefix=$1
work_dir=$2
module_prefix=${3:-"$work_dir/install"}
script_dir=$(cd "$(dirname "$0")" && pwd)
repo_root=$(cd "$script_dir/../../.." && pwd)

module_ninja=${CELLERATOR_MODULE_NINJA:-ninja}
module_cxx=${CELLERATOR_MODULE_CXX:-clang++-18}
module_scanner=${CELLERATOR_MODULE_SCANNER:-clang-scan-deps}
test_cuda=${CELLERATOR_NN_MODULES_TEST_CUDA:-OFF}
torch_lib_dir=${CELLERATOR_TORCH_LIB_DIR:-}
cuda_compiler=${CELLERATOR_CUDA_COMPILER:-/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/cuda/12.9/bin/nvcc}
cuda_host_compiler=${CELLERATOR_CUDA_HOST_COMPILER:-/usr/bin/g++-12}
cuda_toolkit_root=${CELLERATOR_CUDA_TOOLKIT_ROOT:-}
if [[ "$test_cuda" == "ON" && -z "$cuda_toolkit_root" ]]; then
    cuda_toolkit_root=${cuda_compiler%/bin/nvcc}
fi
if [[ -z "$torch_lib_dir" && -n "${CELLERATOR_TORCH_DIR:-}" ]]; then
    torch_lib_dir=$(cd "$CELLERATOR_TORCH_DIR/../../.." && pwd)/lib
fi

common_args=(
    -G Ninja
    "-DCMAKE_MAKE_PROGRAM=$module_ninja"
    "-DCMAKE_CXX_COMPILER=$module_cxx"
    "-DCMAKE_CXX_COMPILER_CLANG_SCAN_DEPS=$module_scanner"
)
if [[ -n "${CELLERATOR_TORCH_DIR:-}" ]]; then
    common_args+=("-DTorch_DIR=$CELLERATOR_TORCH_DIR")
fi
if [[ -n "$cuda_toolkit_root" ]]; then
    common_args+=(
        "-DCUDAToolkit_ROOT=$cuda_toolkit_root"
        "-DCUDA_TOOLKIT_ROOT_DIR=$cuda_toolkit_root"
        "-DCUDA_NVCC_EXECUTABLE=$cuda_toolkit_root/bin/nvcc"
        "-DCMAKE_CUDA_COMPILER=$cuda_compiler"
        "-DCMAKE_CUDA_HOST_COMPILER=$cuda_host_compiler"
    )
fi

cmake -S "$repo_root/modules/native" -B "$work_dir/module-package" \
    "${common_args[@]}" \
    "-DCMAKE_PREFIX_PATH=$cellerator_prefix" \
    "-DCMAKE_INSTALL_PREFIX=$module_prefix"
cmake --build "$work_dir/module-package" --verbose
cmake --install "$work_dir/module-package"

consumer_args=(
    "${common_args[@]}"
    "-DCMAKE_PREFIX_PATH=$module_prefix;$cellerator_prefix"
    "-DCELLERATOR_NN_MODULES_TEST_CUDA=$test_cuda"
)
if [[ "$test_cuda" == "ON" ]]; then
    consumer_args+=(
        "-DCMAKE_CUDA_COMPILER=$cuda_compiler"
        "-DCMAKE_CUDA_HOST_COMPILER=$cuda_host_compiler"
    )
fi

cmake -S "$script_dir" -B "$work_dir/consumer" "${consumer_args[@]}"
cmake --build "$work_dir/consumer" --verbose
if [[ -n "$torch_lib_dir" ]]; then
    LD_LIBRARY_PATH="$torch_lib_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
        ctest --test-dir "$work_dir/consumer" --output-on-failure
else
    ctest --test-dir "$work_dir/consumer" --output-on-failure
fi
