#!/bin/sh
set -eu
mkdir -p build-is1-lowering
cuda_root=/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/cuda/12.9
g++-12 -std=c++20 -Wall -Wextra -pedantic -Iinclude -I"$cuda_root/include" tests/substrate/lowering/guard_test.cc -o build-is1-lowering/guard_test
build-is1-lowering/guard_test
printf "%s\n" "host guard and support tests PASS" > build-is1-lowering/host_gate.log
"$cuda_root/bin/nvcc" -std=c++20 -ccbin g++-12 -arch=sm_70 -O2 --fmad=false -Xptxas=-v -Iinclude src/compute/architecture/providers/nvidia/sm70/substrate/scaled_tanh.cu tests/substrate/lowering/scaled_tanh_gate.cu -o build-is1-lowering/scaled_tanh_gate 2>build-is1-lowering/compile.log
"$cuda_root/bin/cuobjdump" --dump-sass build-is1-lowering/scaled_tanh_gate > build-is1-lowering/scaled_tanh.sass
