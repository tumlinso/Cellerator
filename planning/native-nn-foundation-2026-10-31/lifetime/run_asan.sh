#!/usr/bin/env bash
set -euo pipefail

repo=/home/tumlinson/Cellerator
build=/tmp/cellerator-native-nn-cmake
torch=/home/tumlinson/Software/miniconda3/lib/python3.13/site-packages/torch
cuda=/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/cuda/12.9
output=/tmp/cellerator-native-nn-lifetime-asan
lifetime="$repo/planning/native-nn-foundation-2026-10-31/lifetime"

common=(
  -std=gnu++20 -g -O1 -fno-omit-frame-pointer -fsanitize=address
  -D_GLIBCXX_USE_CXX11_ABI=1
  -DUSE_C10D_GLOO -DUSE_C10D_NCCL -DUSE_DISTRIBUTED -DUSE_RPC -DUSE_TENSORPIPE
  -I"$build/generated" -I"$repo/include"
  -isystem "$torch/include" -isystem "$torch/include/torch/csrc/api/include"
  -isystem "$cuda/include"
)

link=(
  -L/lib/intel64 -L/lib/intel64_win -L/lib/win-x64
  -Wl,-rpath,"$torch/lib:$cuda/lib64:/home/tumlinson/Software/miniconda3/lib:$build"
  "$build/libcellerator_nn.a"
  "$torch/lib/libtorch.so"
  -Wl,--no-as-needed,"$torch/lib/libtorch_cpu.so" -Wl,--as-needed
  -Wl,--no-as-needed,"$torch/lib/libtorch_cuda.so" -Wl,--as-needed
  "$torch/lib/libc10_cuda.so" "$torch/lib/libc10.so"
  "$cuda/lib64/libcudart.so"
  -Wl,--no-as-needed,"$torch/lib/libtorch.so" -Wl,--as-needed
  "$build/libcellerator_product2.so"
  /home/tumlinson/Software/miniconda3/lib/libcrypto.so
)

for test_source in \
  "$lifetime/query_view_repro.cc" \
  "$lifetime/conditioned_view_repro.cc" \
  "$lifetime/full_gather_repro.cc" \
  "$repo/tests/nn/native/query_readouts_test.cc"
do
  test_name=$(basename "$test_source" .cc)
  binary="$output-$test_name"
  g++-12 "${common[@]}" "$test_source" -o "$binary" "${link[@]}"
  ASAN_OPTIONS=detect_leaks=0:halt_on_error=1 "$binary"
done
