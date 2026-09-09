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

B03 leaves compiler selection to CMake's toolchain files, `CXX`, `CUDACXX`,
`CUDAHOSTCXX`, and explicit `CMAKE_*_COMPILER` cache inputs. Host consumers require
C++20; compiled owner libraries retain C++17. sm70 remains the existing default
when no architecture is supplied. Qualification on this machine explicitly used
CUDA 12.9.86, g++-12 as CUDA host compiler, and `CMAKE_CUDA_ARCHITECTURES=70`.
No CUDA 13 or V100 FP8 arithmetic is required or claimed. The B03 test independently
compiles and executes a C++20 consumer, checks explicit compiler preservation,
and verifies that an unavailable compiler produces a configuration error.

B04 exposes `cellerator_nf1_host_correctness`, which builds and runs the real
T01/B01/B02 executables. The `ce_nf1_b04` test independently configures minimal
and retained RU1 host builds and builds/runs `ru1_calculus` and `ru1_reference`.
Requesting retained suites together with the minimal-only option fails clearly,
rather than silently excluding a requested regression. CUDA semantic-spine
examples retain their default availability and can explicitly be disabled with
`CELLERATOR_BUILD_SEMANTIC_SPINE_EXAMPLES=OFF`.

B05 generates `CelleratorNativeFoundationConfig.cmake` in the producer build.
A separate project uses `find_package(CelleratorNativeFoundation CONFIG REQUIRED)`
with `CelleratorNativeFoundation_DIR` pointing there and links
`Cellerator::native_foundation`. Its source and build directories may be anywhere.
The build-tree package refers to the producer's compiled archives and source
headers; moving those requires reconfiguration. It is not an installed SDK or a
self-contained redistributable binary package. The adjacent dependency manifest
records the configure-time Git revision and hashes all public headers plus each
exported compiled source. Final qualification must reconfigure after committing.
`ce_nf1_b05` builds the producer independently, then builds and executes a fresh
external consumer using only the package, supported headers, and imported targets.
