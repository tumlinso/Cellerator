# Tokenization build evidence

Builds were performed in isolated `build/tokenization-*` directories.

## CPU C++ and Python

The C++ build used extracted PyTorch 2.7.0+cpu at
`/tmp/cellerator-tokenization-cpu-torch`, with tokenization enabled and CUDA
disabled. Focused CTest result: 3/3 passed (`numerics`, `core`, `append`).

```sh
cmake -S . -B build/tokenization-cpu-3 \
  -DCELLERATOR_ENABLE_CUDA=OFF -DCELLERATOR_ENABLE_TOKENIZATION=ON \
  -DCELLERATOR_ENABLE_PYTHON=OFF -DCELLERATOR_BUILD_PRODUCT2=OFF \
  -DCELLERATOR_BUILD_TESTS=ON \
  -DTorch_DIR=/tmp/cellerator-tokenization-cpu-torch/torch/share/cmake/Torch \
  -DCaffe2_DIR=/tmp/cellerator-tokenization-cpu-torch/torch/share/cmake/Caffe2
cmake --build build/tokenization-cpu-3 --target \
  cellerator_tokenization_numerics_test \
  cellerator_tokenization_core_test \
  cellerator_tokenization_append_test -j4
ctest --test-dir build/tokenization-cpu-3 --output-on-failure \
  -R '^cellerator_tokenization_'
```

The Python 3.13 extension was built against the same CPU Torch package. The
public Python suite passed 25 tests (20 passed, 5 CUDA-only tests skipped).

```sh
PYTHONPATH=/tmp/cellerator-tokenization-cpu-torch \
  cmake --build build/tokenization-python-cpu \
  --target cellerator_python_tokenization -j2
PYTHONPATH=/tmp/cellerator-tokenization-cpu-torch:build/tokenization-python-cpu/python \
  /home/tumlinson/Software/miniconda3/bin/python \
  tests/tokenization/test_tokenization.py
```

An installed C++ consumer found `Cellerator::tokenization`, built, ran, and
resolved `libtorch`, `libtorch_cpu`, and `libc10` from the selected CPU package.
The default-off CUDA/tokenization configuration also configured successfully;
it exposed no tokenization targets.

## CUDA build and wheel

The final CUDA build used Python Torch 2.7.0+cu126, HPC SDK nvcc 12.9,
`g++-12`, SM70, lineinfo, and disabled hardware probing. The full
`cellerator_python_extensions` target, including `_native` and
`_tokenization`, built successfully. CUDA code was compiled only; no GPU code
was executed in this build worker.

```sh
cmake -S . -B build/tokenization-cuda-python-4 \
  -DCELLERATOR_ENABLE_CUDA=ON -DCELLERATOR_ENABLE_TOKENIZATION=ON \
  -DCELLERATOR_ENABLE_PYTHON=ON -DCELLERATOR_BUILD_PRODUCT2=ON \
  -DCELLERATOR_BUILD_TESTS=OFF -DCELLERATOR_ENABLE_HARDWARE_PROBE=OFF \
  -DCELLERATOR_AUTO_DETECT_CUDA_ARCHITECTURES=OFF \
  -DCMAKE_CUDA_COMPILER=/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/cuda/12.9/bin/nvcc \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-12 \
  -DCMAKE_CUDA_ARCHITECTURES=70 -DCELLERATOR_ENABLE_CUDA_LINEINFO=ON \
  -DTorch_DIR=/home/tumlinson/Software/miniconda3/lib/python3.13/site-packages/torch/share/cmake/Torch \
  -Dpybind11_DIR=/home/tumlinson/Software/miniconda3/lib/python3.13/site-packages/pybind11/share/cmake/pybind11 \
  -DPython_EXECUTABLE=/home/tumlinson/Software/miniconda3/bin/python
cmake --build build/tokenization-cuda-python-4 \
  --target cellerator_python_extensions -j2
```

Wheel:

- Path: `/tmp/cellerator-tokenization-wheel-final5/cellerator-0.1.0-cp313-cp313-linux_x86_64.whl`
- Size: 540,773 bytes
- SHA256: `994a3d785c16e7b43b7a31ece6b69fb757fe06713c95fc22929cd107e60ba496`
- Install command for the selected Python (run by the parent task owner):

```sh
/home/tumlinson/Software/miniconda3/bin/python -m pip install --no-deps \
  /tmp/cellerator-tokenization-wheel-final5/cellerator-0.1.0-cp313-cp313-linux_x86_64.whl
```

The final Geneformer profile exposes `compute_type=None` as its default
precision policy and accepts explicit `"float32"` or `"float64"`. The rebuilt
CPU extension passed the full Python suite and a default/explicit-FP64 call
smoke check after that API update.
