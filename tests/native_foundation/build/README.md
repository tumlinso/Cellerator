# NF1 source-linked host build

The bounded host target `Cellerator::native_foundation` composes the existing
operation-core-v2 schema and prepared-program-v2 compiled owners, plus their
C++20 contract header. It does not introduce another executor or device provider.
See `docs/architecture.qmd` and the NF1 package for architectural ownership.

```sh
cmake -S . -B build-host -DCELLERATOR_ENABLE_CUDA=OFF \
  -DCELLERATOR_NATIVE_FOUNDATION_ONLY=ON \
  -DCELLERATOR_BUILD_NATIVE_FOUNDATION_TESTS=ON
cmake --build build-host
ctest --test-dir build-host --output-on-failure --no-tests=error
```

An enclosing CMake consumer can set these cache options, call `add_subdirectory`
on this repository, then link `Cellerator::native_foundation`. No CUDA, Torch,
CellShard, Baseplane or compiler-model discovery runs in this minimal slice.
The ordinary build continues to expose the retained compiler and CUDA targets.

`ce_nf1_b01` executes an actual affine host callback through the existing prepared
program executor, checks exact outputs and rejects missing bindings/invalid
stage identities. T01 remains the independent nonlinear FP64 numerical oracle.
These tests do not qualify device execution, install/export packaging, or a full
native numerical provider.

`CELLERATOR_BUILD_NATIVE_NUMERIC=AUTO` includes the N lane's production/test
fragments once present. Explicit `ON` fails configuration if a fragment is
missing; `OFF` excludes that capability. Empty CTest inventory never qualifies.

B02 rehomes the retained relation semantic/calculus and segment host numerical
owners and gate validator into compiled targets. Use
`cellerator_link_native_foundation(consumer)` after declaring source files to
link them and reject direct private `.cu`, `.cc`, or `.cpp` inclusions.
`cellerator_link_native_cuda(consumer)` additionally requires and links the real
`prepared_relation_cuda`, `relation_algebra` (segment and gate device kernels),
and `runtime` (value readiness) owners. Missing device targets fail configuration;
the host slice does not substitute for them. Declare consumer source files
before calling the helper; it checks these direct sources, not preprocessor
macro expansion or all transitively included headers.

The B02 executable reuses the retained segment reduction test, including empty
segments, extrema and paired moments. Its companion boundary test configures an
external project and verifies rejection of a private CUDA implementation include.
