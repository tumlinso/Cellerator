# Native NN CPU build and install consumer

Configure and build the optional C++20 `Cellerator::nn` target without the
repository CUDA backend or Python bindings:

```sh
cmake -S . -B /tmp/cellerator-native-nn-cmake \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++-12 \
  -DCELLERATOR_ENABLE_CUDA=OFF \
  -DCELLERATOR_ENABLE_NN=ON \
  -DCELLERATOR_ENABLE_PYTHON=OFF \
  -DCELLERATOR_ENABLE_TORCH=OFF \
  -DCELLERATOR_BUILD_PRODUCT2=ON \
  -DCELLERATOR_BUILD_TESTS=OFF \
  -DCELLERATOR_BUILD_NN_TESTS=OFF \
  -DTorch_DIR=/home/tumlinson/Software/miniconda3/lib/python3.13/site-packages/torch/share/cmake/Torch \
  -DCMAKE_PREFIX_PATH=/home/tumlinson/Software/miniconda3 \
  -DCUDA_TOOLKIT_ROOT_DIR=/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/cuda/12.9 \
  -DCUDA_NVCC_EXECUTABLE=/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/cuda/12.9/bin/nvcc \
  -DCUDAToolkit_ROOT=/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/cuda/12.9
cmake --build /tmp/cellerator-native-nn-cmake --target cellerator_nn -j2
```

The explicit toolkit paths are needed here because the selected Torch 2.7 CUDA
package makes Caffe2 validate its CUDA headers during `find_package(Torch)`.
`CELLERATOR_ENABLE_CUDA=OFF` still avoids Cellerator CUDA targets; this check
does not run any device code.

Stage an install under a fresh `/tmp` directory, compile an independent
`find_package(Cellerator COMPONENTS nn)` consumer, and run its CPU autograd plus
Torch SGD path with:

```sh
planning/native-nn-foundation-2026-10-31/build/run_cpu_installed_consumer.sh \
  /tmp/cellerator-native-nn-cmake
```

The script prints and preserves its temporary install prefix and consumer build
directory for inspection.
