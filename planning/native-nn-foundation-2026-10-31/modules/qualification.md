# Native `cellerator.nn` C++20 module qualification

## Surface and packaging

The optional host module `cellerator.nn` imports the canonical declarations
from `<Cellerator/nn.hh>` in a module global fragment and exports selected
`using` declarations. These remain the existing C++ types and entities in
`cellerator::nn`, `cellerator::biology`, `cellerator::nn::ops`, and the ATen
names used by their signatures. The add-on package exports
`CelleratorNNModules::nn` and finds the installed dependency through
`Cellerator::nn`; it does not change the main Cellerator compiler toolchain.
The interface imports neither `std` nor third-party header units. Third-party
headers remain ordinary includes in the consumer.

The installed CMake module target re-scans the installed module source for its
consumer. The CMake 3.28 imported-target metadata did not propagate the
dependency's include directories to the module scanner, so the package config
sets `IMPORTED_CXX_MODULES_INCLUDE_DIRECTORIES` from
`Cellerator::nn`'s public include directories. The resulting scanner command
used the installed Cellerator and Torch include roots.

## CPU consumer

The external consumer was configured against the canonical installed package
at `/tmp/cellerator-native-nn-consumer.peeU8C/install`, plus the separately
installed `CelleratorNNModules` add-on. It used `find_package(Cellerator
COMPONENTS nn)` and imported `cellerator.nn` from a normal C++20 translation
unit. The consumer:

- aligned string biological feature IDs with `FeatureBinding` and
  `OccurrenceCoordinates`;
- passed aligned model indices through `FeatureEncoder`;
- composed a user-defined two-layer C++ model from native `Linear` components;
- calculated an ATen MSE loss, ran backward, and updated the owned parameters
  with `torch::optim::SGD`;
- passed the resulting `Parameters` and `at::Tensor` across a separate
  translation unit that includes `<Cellerator/nn.hh>`;
- compiled a second importing translation unit with no preceding headers;
  it constructs `Parameters` and verifies that `at::Tensor` and `at::Device`
  are complete through the module import alone.

Validation used CMake 3.28.3, Clang 18.1.3, Ninja 1.11.1, and the matching
`clang-scan-deps` 18.1.3. Ninja and the scanner were staged under
`/tmp/cellerator-module-tools`; no environment-wide tools were installed.
Torch was 2.7.0+cu126. The module package scan,
build, install, external CMake configure, module import build, link against
the installed `Cellerator::nn` closure, and runtime consumer all passed. CTest
reported `1/1` passed. The reproducible driver is
[`run_qualification.sh`](../../../tests/nn/modules/run_qualification.sh); the
CPU run used:

```sh
CELLERATOR_MODULE_NINJA=/tmp/cellerator-module-tools/ninja/usr/bin/ninja \
CELLERATOR_MODULE_CXX=/usr/bin/clang++-18 \
CELLERATOR_MODULE_SCANNER=/tmp/cellerator-module-tools/clang/usr/lib/llvm-18/bin/clang-scan-deps \
CELLERATOR_TORCH_DIR=/home/tumlinson/Software/miniconda3/lib/python3.13/site-packages/torch/share/cmake/Torch \
CELLERATOR_TORCH_LIB_DIR=/home/tumlinson/Software/miniconda3/lib/python3.13/site-packages/torch/lib \
CELLERATOR_CUDA_TOOLKIT_ROOT=/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/cuda/12.9 \
tests/nn/modules/run_qualification.sh \
  /tmp/cellerator-native-nn-consumer.mQYDYs/install \
  /tmp/cellerator-nn-module-cpu-current-source
```

This CMake 3.28/Ninja configuration did not accept two distinct executable
targets importing the same installed module target: generation emitted a
duplicate Ninja rule for CMake's synthetic imported-module target. The
qualified project keeps its import-only and training translation units in one
consumer executable. Multiple importing translation units within that target
passed; a multi-executable project shape remains unqualified on this CMake
version.

The CMake configuration emitted nonfatal Torch/Caffe2 warnings about NVRTC
hash discovery and missing NVTX3. No CUDA kernel was launched. The CPU run
qualifies this consumer and configuration only; it does not establish
compiler-portable module artifacts or coverage of every exported declaration.

## NVCC/header bridge and CUDA linkage

The external consumer was also configured against the current-source CUDA
install at `/tmp/cellerator-native-nn-cuda-current-install` with
`CELLERATOR_NN_MODULES_TEST_CUDA=ON`. CMake imported
`Cellerator::nn_cuda` through the `Cellerator::nn` dependency closure. Clang
compiled the importing and ordinary header translation units; NVCC 12.9.86
compiled the `.cu` bridge with `/usr/bin/g++-12` as its host compiler. The
generated NVCC rule confirmed `-ccbin=/usr/bin/g++-12`. The `.cu` bridge
includes only `<Cellerator/nn.hh>` and accepts the same `Parameters`,
`at::Tensor`, and `at::Device` values as the importing Clang translation
unit. It does not parse module syntax. The importing consumer takes the
address of `cellerator::nn::ops::grouped_linear_cuda` through a volatile
function pointer, forcing static-library symbol resolution without invoking
the operation. The linked consumer ran its CPU training path; CTest reported
`1/1` passed. No CUDA kernel was launched.

The CUDA qualification used:

```sh
CELLERATOR_MODULE_NINJA=/tmp/cellerator-module-tools/ninja/usr/bin/ninja \
CELLERATOR_MODULE_CXX=/usr/bin/clang++-18 \
CELLERATOR_MODULE_SCANNER=/tmp/cellerator-module-tools/clang/usr/lib/llvm-18/bin/clang-scan-deps \
CELLERATOR_TORCH_DIR=/home/tumlinson/Software/miniconda3/lib/python3.13/site-packages/torch/share/cmake/Torch \
CELLERATOR_TORCH_LIB_DIR=/home/tumlinson/Software/miniconda3/lib/python3.13/site-packages/torch/lib \
CELLERATOR_CUDA_TOOLKIT_ROOT=/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/cuda/12.9 \
CELLERATOR_CUDA_COMPILER=/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/cuda/12.9/bin/nvcc \
CELLERATOR_CUDA_HOST_COMPILER=/usr/bin/g++-12 \
CELLERATOR_NN_MODULES_TEST_CUDA=ON \
tests/nn/modules/run_qualification.sh \
  /tmp/cellerator-native-nn-cuda-current-install \
  /tmp/cellerator-nn-module-cuda-current-prefix
```

Torch/Caffe2 emitted nonfatal NVRTC hash and NVTX3 discovery warnings during
configuration. No CUDA device code ran. This qualifies the CMake 3.28.3,
Clang 18, Ninja, NVCC 12.9, and GNU 12 path only; other module compiler and
generator combinations were not exercised.
