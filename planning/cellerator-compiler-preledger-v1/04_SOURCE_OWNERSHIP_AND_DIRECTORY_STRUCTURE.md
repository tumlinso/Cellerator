# Source ownership and proposed directory structure

## Proposed tree

```text
include/Cellerator/
    compiler/
        driver/
        frontend/
            source/
            parser/
            cxx/
        ast/
        sema/
        ir/
            common/
            text/
            semantic/
            planning/
            realization/
        profile/
        discovery/
        composition/
        program/
        planning/
        reflection/
        pass/
        diagnostics/
        lto/
        backend/
            cpu/
            nvcc/
            nvptx/
        tooling/
        api/
    sdk/
    compiler.hh
    runtime.hh
    Cellerator.hh

src/compiler/
    driver/
    frontend/
    ast/
    sema/
    ir/
    profile/
    discovery/
    composition/
    program/
    planning/
    reflection/
    pass/
    diagnostics/
    lto/
    backend/
    tooling/
    api/

tools/
    cellerator/
    celleratord/

stdlib/
    cellerator/
        core.cell
        domain/
        relation/
        operation/
        profile/
        planning/
        reflection/
        ir/

profiles/reference/
tests/compiler/
tests/celleratord/
tests/sdk/
tests/package/
bench/compiler/
cmake/compiler/
cmake/package/
docs/compiler/
docs/language/
```

## Build ownership

The root project is planned as `LANGUAGES CXX`; CUDA is enabled conditionally through `CELLERATOR_ENABLE_CUDA=AUTO|ON|OFF`.

Compiler core, CEIR, profiles, diagnostics, and celleratord protocol code do not link CUDA.

Backend/provider targets own CUDA language activation and CUDAToolkit dependencies.

Existing runtime/provider targets remain available and may retain C++17/CUDA17. New compiler/tooling code uses a modern implementation standard, initially C++23 where supported. The downstream source mode follows driver arguments.

## Public library ownership

The installed SDK exposes component libraries and a convenience umbrella:

- compiler core/frontend/IR/profile/planning/realization/backend/diagnostics;
- runtime/execution/geometry/planner/provider components;
- `libCellerator` convenience facade.

`tools/cellerator/main.cc` and `tools/celleratord/main.cc` remain thin.

## Central integration-only files

Leaf work is isolated. The following are integration-owned:

- root and subsystem CMake aggregators;
- public umbrella headers;
- canonical grammar/dialect/backend/pass registries;
- generated compiler/resource manifests;
- standard-library manifest;
- package exports;
- normative documentation synchronization;
- `.gitmodules` and the `components/CellShard` gitlink.

## Generated files

Generated parser tables, dialect manifests, version headers, embedded resources, and backend manifests live in the build tree. Their source schemas/generators are checked in and reproducible.

## Current-to-proposed map

The machine-readable migration inventory is `inventories/jbc_source_migration.csv`. The final architecture does not delete the current execution, geometry, planner, or runtime subtrees. It adds compiler-facing layers that adapt and expose them through CEIR and libCellerator.
