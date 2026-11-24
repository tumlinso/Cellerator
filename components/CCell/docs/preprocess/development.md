# Development

The CCell component keeps preprocessing in `src/preprocess/` and its tests in
`tests/preprocess/`. The Cellerator root build owns the module target, tests,
installation, and optional qualification targets. Use the
Clang, clang-scan-deps, CMake, and Ninja versions in the
[root build instructions](../../README.md).

The numerical module accepts native Cellerator views. Consumers keep input
storage alive, provision output and workspace, and control parallelism.
Checked calls report status and required capacity. Trusted calls use the
documented preconditions. The reference allocates no memory during sizing,
validation, preparation, or execution.

With the reference Python environment installed, run the independent Scanpy
1.11.5 comparison from the repository root:

~~~sh
python -m pip install -r components/CCell/tests/preprocess/requirements-reference.txt
cmake -S . -B build/debug -G Ninja \
  -DCMAKE_CXX_COMPILER=clang++-18 -DCELLERATOR_ENABLE_CUDA=OFF \
  -DCELLERATOR_ENABLE_ANNDATA=OFF -DCCELL_BUILD_QUALIFICATION=ON
cmake --build build/debug --target ccell_preprocess_interop_driver --parallel 2
(cd components/CCell && python tools/preprocess/qualify.py \
  --driver ../../build/debug/components/CCell/src/preprocess/ccell_preprocess_interop_driver)
~~~

By default, the report is written to
`build/qualification/preprocess-reference.json`; use `--output` for another
disposable path. Historical reports under `tests/preprocess/results/` are
inputs to provenance and must not be overwritten.

The native runtime observation uses the same root option:

~~~sh
cmake --build build/debug --target ccell_preprocess_benchmark --parallel 2
(cd components/CCell && python tools/preprocess/measure_runtime.py \
  --driver ../../build/debug/components/CCell/src/preprocess/ccell_preprocess_benchmark)
~~~

Its report defaults to `build/qualification/preprocess-runtime.json`. Each
subprocess is bounded to 30 seconds; reports characterize the stated inputs
and toolchain and do not establish universal speed or biological validity.

Older standalone configure, package, qualification, and build measurements
remain historical evidence. Root CMake and the current source tree are the
authority for maintained builds.
