# Cellerator Agent Guide

## Scope

Cellerator is now the umbrella workspace for the core genomics GPU stack. It
owns workspace layout, component submodule pointers, cross-component build
coordination, root documentation, shared fixture scaffolding, and the minimal
foundation code required by child components.

Implementation ownership stays sharp:

- `components/Baseplane/` owns low-level sequence bit primitives, CPU/SIMD
  backends, and CUDA sequence kernels for compact DNA/RNA representations.
- `components/CellShard/` owns storage, metadata, pack generation, runtime
  delivery, and layout staging.
- `include/Cellerator/` and `src/foundation/` own only umbrella foundation
  primitives required by components: sparse layout ABI, CUDA runtime substrate,
  distributed runtime declarations, and matrix conversion helpers.
- `Archive/current-cellerator/` is source-only historical material. Do not add
  new implementation there except to preserve archive build metadata.

Do not add `cudaBioTypes` to this workspace unless the project direction
changes explicitly.

## Working In Components

Before changing files inside `components/Baseplane/` or `components/CellShard/`,
read that component's own `AGENTS.md`. Component guidance is authoritative for
that project's implementation style, tests, and package surface.

Keep hot reusable math in the owning compute/runtime layer. Workflow or session
layers must not duplicate GPU reductions, dense adds, metric packets, sparse
transforms, normalization/stat operators, fleet collectives, or scratch
mechanics.

## Root Changes

Root-level changes should normally be limited to:

- umbrella CMake and package coordination
- component submodule declarations
- `README.md`, `AGENTS.md`, `docs/`, and validation notes
- shared fixture scaffolding under `data/test`
- helper scripts under `scripts/`
- `todos.md`, `todo-status.md`, and workstream ledgers

Do not put new model, preprocessing, storage, ingest, or sequence implementation
code directly in the umbrella root unless it is part of the foundation slice
that a component already requires.

## Build And Test Coordination

Run the umbrella build from the Cellerator root:

```bash
cmake -S . -B build
cmake --build build -j 4
```

Focused validation should include:

```bash
cmake --build build --target baseplaneDna2Test cellShardInspectPackageTest celleratorFoundationRuntimeTest -j 4
./build/baseplaneDna2Test
./build/celleratorFoundationRuntimeTest
```

Benchmark and profiler runs must be serialized across workers. Use the relevant
component benchmark mutex when one exists and record exact commands plus
hardware/toolchain context when timings matter.

## Git Hygiene

Prefer committing and pushing component submodule changes before updating the
umbrella pointer. For cross-component work:

1. Make and verify the implementation change inside the owning component.
2. Commit and push the component change.
3. Update the Cellerator umbrella submodule pointer.
4. Commit and push the Cellerator umbrella coordination change.

Do not use the umbrella root to hide uncommitted work inside a component.
