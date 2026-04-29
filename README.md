# Cellerator

Cellerator is a GPU-oriented math, compute, and model package for large sparse single-cell datasets in the CellStack family.

It consumes CellShard as an external dependency for sparse matrix/runtime ABI types. Cellerator owns math over those matrices: reusable sparse compute, model code, trajectory logic, quantized kernels, and Torch-facing boundaries. Storage, ingest, preprocessing workflow policy, neighbor-caller orchestration, and biological semantic validation live in sibling CellStack projects or installed packages.

This codebase is built for explicit low-level control rather than a high-level workflow API. The durable scope contract is in `scope.md`: Cellerator owns sparse biological ML operators, training primitives, model components, distributed sparse execution, and explicit Torch interop. It does not replace Torch, AnnData, Scanpy, or CellShard, and the future compiled `.cellerator` model format is out of scope for now.

Surfaces that are temporarily inside this repo but do not belong in Cellerator long term are tracked in `out_of_scope_inventory.md`.

## What Is Here

- `src/compute/`: reusable sparse compute, neighbors, autograd, and custom-op building blocks
- `src/models/`: libtorch model workflows
- `src/trajectory/`: trajectory scoring and assembly
- `src/quantized/`: quantized sparse runtime
- `src/torch/`: explicit CellShard-to-Torch boundary
- `tests/`: compile and runtime checks
- `bench/`: benchmark binaries

Expected local source checkout layout:

```text
cellstack/
├── CellShard/
├── Cellerator/
├── CellShardPreprocess/
└── NeighborCaller/
```

## Source Layout

Cellerator is mainly a header-first C++/CUDA codebase. A lot of the reusable surface lives in `.hh` and `.cuh` headers, with `.cc` and `.cu` translation units used for tests, binaries, and heavier implementation boundaries.

Common file roles:

- `.hh`: C++ headers for public or reusable host-side interfaces
- `.cuh`: CUDA-aware headers used by device-facing or mixed host/device code
- `.cc`: C++ source files
- `.cu`: CUDA source files

Naming is mostly `snake_case` for files, functions, variables, and structs.

The main source areas are:

- `include/Cellerator/`: canonical public headers for in-repo callers
- `src/compute/`: sparse ML math over CellShard matrices
- `src/compute/core/`: shared low-level host/device utilities
- `src/compute/ml/autograd/`: sparse autograd contexts, scratch, and kernels
- `src/compute/neighbors/`: exact-search and scoring math only; index/query caller policy is external
- `src/models/`: header-first model modules, typically split into `*_dataloader.hh`, `*_model.hh`, `*_train.hh`, and `*_infer.hh`
- `src/quantized/`: quantized sparse layouts, accessors, and kernels
- `src/torch/`: explicit Torch interop boundary

Public in-repo includes should use `#include <Cellerator/...>` rather than reaching into `src/` directly.

## Quick Start

Configure and build:

```bash
cmake -S . -B build
cmake --build build -j 4
```

CMake resolves CellShard in this order:

1. `-DCELLERATOR_CELLSHARD_SOURCE_DIR=/path/to/CellShard`
2. sibling `../CellShard`
3. `find_package(CellShard CONFIG REQUIRED)`

CUDA mode defaults to `generic`. Use `-DCELLERATOR_CUDA_MODE=native` for the
host-specific fast path and `-DCELLERATOR_CUDA_MODE=native-extreme` for the
separate Volta-only/PTX-heavy path.

The Cellerator build is accelerator-oriented and requires CUDA at configure time.

Torch-enabled builds prefer the source-built libtorch installation under `/usr/local/share/cmake/Torch`.

If libtorch is not available, disable Torch models:

```bash
cmake -S . -B build -DCELLERATOR_ENABLE_TORCH_MODELS=OFF
cmake --build build -j 4
```

Only set `Torch_DIR` or `LIBTORCH_PATH` if you are intentionally overriding the default libtorch.

## Running Things

`ctest` is not configured. Run the built binaries directly.

Examples:

```bash
./build/exactSearchRuntimeTest
./build/computeAutogradRuntimeTest
./build/quantizeModelTest
```

Useful build targets include:

- `quantizedMatrixTest`
- `trajectoryCompileTest`
- `trajectoryRuntimeTest`
- `exactSearchRuntimeTest`
- `computeAutogradRuntimeTest`
- `developmentalTimeCompileTest`
- `denseReduceCompileTest`
- `quantizeModelTest`
- `torchBindingsCompileTest`
- `modelCustomOpsTest`

## Repository Notes

- Cellerator is performance-oriented and currently tuned around Volta / V100-class assumptions.
- CUDA mode selection is explicit: `generic` is the default topology-agnostic path, while `native` and `native-extreme` unlock the host-specific V100 ordering only after runtime discovery confirms that topology.
- Blocked-ELL is the preferred native sparse execution layout for Cellerator hot paths; CSR/compressed remains an explicit fallback or interop representation where a surface still requires it.
- Cellerator owns model-facing numerical math over CellShard matrices. Data handling, source ingest, preprocessing policy, neighbor-caller policy, and biological semantic validation stay outside this package.
- `docs/` hosts Cellerator architecture and local-only workflow notes; it is not part of the default build or packaging surface.
- `docs/quantized_transfer_architecture.qmd` records the current Volta-oriented report for quantized sparse transport, decode cost, and tensor-op-oriented follow-up work.

## Where To Read Next

- `scope.md`: canonical Cellerator ownership and out-of-scope boundary
- `out_of_scope_inventory.md`: advisory queue for code that should move out or be reclassified
- `planning_strategy.md`: pickup guide for operator-first sparse biological ML planning
- `AGENTS.md`: contributor rules, codebase conventions, and repository-specific guidance
- `optimization.md`: performance notes and bottleneck analysis
- `pointer_migration_plan.md`: pointer-first migration policy for hot paths
- `todos.md`: active work ledger when substantial repo work is in progress
