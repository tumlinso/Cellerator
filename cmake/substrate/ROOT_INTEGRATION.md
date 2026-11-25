# Root build integration

The independent SDK entry point is `cmake -S cmake/substrate`. It selects actual
CE translation units, never sibling private sources. The normal native CUDA and
optional Cellerator Torch adapter builds retain their native owners.

For a root entry point, add this block directly after `project(Cellerator LANGUAGES CXX)`
and before native/compiler/adaptor target creation:

```cmake
option(CELLERATOR_BUILD_SUBSTRATE_SDK "Build the selected substrate SDK" OFF)
if(CELLERATOR_BUILD_SUBSTRATE_SDK)
    include(cmake/substrate/Substrate.cmake)
    return()
endif()
```

This early branch prevents collisions with existing namespace aliases and optional
CUDA/Torch package exports. The component module is a selectable build projection
of the existing source owners; it introduces no alternate evaluator or runtime.
Do not append it after the current NativeFoundation export closure: that closure
uses a different host/compiler package setup and includes CUDA/training dependencies.

For the full native branch, add the accepted leaf owner subdirectories after their
real prerequisites have been created, separately from this standalone SDK projection:

```cmake
add_subdirectory(src/state)
add_subdirectory(src/layout/packing/core) # native_numeric + selected indexed_incidence/full indexed_mechanism + cellpack
add_subdirectory(src/ops/matrix)
add_subdirectory(src/ops/process)
add_subdirectory(src/math/effects)
add_subdirectory(src/execution/prepared_host) # local_differential
```

Their existing include paths need BUILD_INTERFACE/INSTALL_INTERFACE conversion
before inclusion in the legacy native export set. The `effects` component uses
operation-family headers under `Cellerator/ops/`. The selectable SDK exports
merged source, and `CELLERATOR_SUBSTRATE_REQUIRE_INTEGRATED=ON` rejects missing
prerequisites and adds the six integrated leaf target names.

The host SDK `indexed_incidence` is explicitly incidence-only. The unchanged
`evaluators.cc` object references `evaluate_cuda_f32`, so even host evaluator use
requires its CUDA object. Preserve `Cellerator::indexed_mechanism` for its original full owner. MERGE-A
should link host PACK to `Cellerator::indexed_incidence`. The original full indexed/training target remains in the
native CUDA build; this installer neither stubs missing symbols nor copies owners.

The optional `native_numeric_sm70` package component compiles the actual existing
`device_linear.cu`, links `native_numeric` and CUDA runtime, and requires CUDA 12.x,
SM70. It does not export full indexed GPU training or Torch adapters. Execution
qualification uses Device Scheduler.

## Target dependency direction

The SDK selects the same `Cellerator::sequence` target as the normal native
build. The target owns exact packed sequence operations and predicate
preparation. Select it with `find_package(Cellerator CONFIG REQUIRED COMPONENTS
sequence)`. It requires neither CUDA nor a separate sequence package.

The former Moonshot component exports are retired. Consumers select maintained
owners such as `effects`, `frontier`, and `sequence` directly. Original inputs
and evidence remain available in the
[pre-retirement revision](https://github.com/tumlinso/Cellerator/tree/2c2c222f36c3aa1f7f8bb22d97031002c6420978/experiments).
