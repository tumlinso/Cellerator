# C++20 module interfaces

`cellerator.nn` is an optional host C++20 import facade over the ordinary
`<Cellerator/nn.hh>` API. The module includes the canonical header in its global
module fragment and exports selected `using` declarations for those existing
types and functions. Header, module, and CUDA translation units therefore use
the same declarations, layouts, and ABI. CUDA sources continue to include the
ordinary headers; NVCC does not parse module syntax.

With `CELLERATOR_ENABLE_NN=ON` and `CELLERATOR_ENABLE_MODULES=ON`,
[`cellerator.nn.ccm`](cellerator.nn.ccm) is built and installed with the NN
component. Both header and import consumers link `Cellerator::nn`. Installed
consumers request `find_package(Cellerator CONFIG REQUIRED COMPONENTS nn)`.
Enable `CXX_SCAN_FOR_MODULES` on the target or C++ sources that use
`import cellerator.nn;`.

Module builds require CMake 3.28 or newer, Ninja, and a Clang compiler with a
matching `clang-scan-deps` executable.

The module-enabled package installs the module source and CMake file-set
metadata. CMake builds the interface with the consumer's compiler when a target
links `Cellerator::nn`, including targets whose own sources use only headers.

The [external qualification](../tests/modules/README.md) uses CMake 3.28,
Clang 18, Ninja, and `clang-scan-deps` 18. Its optional NVCC bridge compiles
against the ordinary headers with NVCC 12.9 and GNU 12 as the host compiler.
