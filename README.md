# Cellerator

Cellerator is a GPU-oriented compute and model repository for large sparse single-cell datasets.

It sits on top of `CellShard`, which handles canonical sparse storage, pack delivery, and distributed execution staging. Cellerator adds ingest, one-pass preprocessing/filtering, reusable sparse compute, model code, trajectory logic, Torch-facing boundaries, and the ncurses workbench.

This codebase is built for explicit low-level control rather than a high-level workflow API, but it now also carries a thin Python wrapper for the dataset-facing workbench path. If you want the storage layer by itself, see `extern/CellShard/`.

## What Is Here

- `src/ingest/`: source readers, MTX conversion, and dataset ingest
- `src/compute/`: reusable sparse compute, preprocess operators, neighbors, and custom-op building blocks
- `src/models/`: libtorch model workflows
- `src/trajectory/`: trajectory scoring and assembly
- `src/workbench/`: keyboard-driven terminal workbench
- `src/quantized/`: quantized sparse runtime
- `src/torch/`: explicit CellShard-to-Torch boundary
- `tests/`: compile and runtime checks
- `bench/`: benchmark binaries
- `extern/CellShard/`: storage and staging submodule used by Cellerator

## Source Layout

Cellerator is mainly a header-first C++/CUDA codebase. A lot of the reusable surface lives in `.hh` and `.cuh` headers, with `.cc` and `.cu` translation units used for tests, binaries, and heavier implementation boundaries.

Common file roles:

- `.hh`: C++ headers for public or reusable host-side interfaces
- `.cuh`: CUDA-aware headers used by device-facing or mixed host/device code
- `.cc`: C++ source files
- `.cu`: CUDA source files

Naming is mostly `snake_case` for files, functions, variables, and structs.

The main source areas are:

- `src/compute/`: low-level compute building blocks and performance-sensitive kernels
- `src/compute/autograd/`: sparse autograd contexts, scratch, and kernels
- `src/compute/neighbors/`: nearest-neighbor and retrieval code
- `src/ingest/`: source-format readers, metadata tables, partition planning, and dataset conversion
- `src/models/`: header-first model modules, typically split into `*_dataloader.hh`, `*_model.hh`, `*_train.hh`, and `*_infer.hh`
- `src/workbench/`: the ncurses workbench application and helpers
- `src/quantized/`: quantized sparse layouts, accessors, and kernels
- `src/torch/`: explicit Torch interop boundary
- `src/legacy/`: older monolithic or transitional code that is not the preferred surface for new work

## Quick Start

Configure and build:

```bash
cmake -S . -B build
cmake --build build -j 4
```

The integrated root build is accelerator-only: it requires CUDA at configure
time and hard-wires the vendored `CellShard` subproject onto its CUDA runtime
surface.

Torch-enabled builds prefer the source-built libtorch installation under `/usr/local/share/cmake/Torch`.

If libtorch is not available, disable Torch models:

```bash
cmake -S . -B build -DCELLERATOR_ENABLE_TORCH_MODELS=OFF
cmake --build build -j 4
```

Only set `Torch_DIR` or `LIBTORCH_PATH` if you are intentionally overriding the default libtorch.

To build the optional Python wrapper from the root build:

```bash
cmake -S . -B build -DCELLERATOR_ENABLE_PYTHON=ON -DCELLERATOR_ENABLE_TORCH_MODELS=OFF -DCELLERATOR_ENABLE_NCURSES_WORKBENCH=OFF
cmake --build build -j 4 --target celleratorPythonCompileTest
```

If you also enable `-DCELLERATOR_ENABLE_CELLSHARD_PYTHON=ON`, the root build exposes a focused `celleratorPythonRuntimeTest` target that imports both build-tree packages and runs an end-to-end smoke check.

## Running Things

`ctest` is not configured. Run the built binaries directly.

Examples:

```bash
./build/cellShardDatasetH5Test
./build/datasetWorkbenchRuntimeTest
./build/forwardNeighborsCompileTest
./build/computeAutogradRuntimeTest
./build/quantizeModelTest
```

Useful build targets include:

- `celleratorWorkbench`
- `celleratorPythonCompileTest`
- `celleratorPythonRuntimeTest` when both Python modules are enabled
- `cellShardDatasetH5Test`
- `datasetIngestCompileTest`
- `datasetWorkbenchRuntimeTest`
- `forwardNeighborsCompileTest`
- `computeAutogradRuntimeTest`
- `developmentalTimeCompileTest`
- `denseReduceCompileTest`
- `quantizeModelTest`
- `torchBindingsCompileTest`
- `modelCustomOpsTest`

## Repository Notes

- Cellerator is performance-oriented and currently tuned around Volta / V100-class assumptions.
- Blocked-ELL is the preferred sparse execution and persistence layout; CSR/compressed remains available where needed.
- Current MTX-dataset ingest is bounded-memory and SSD-aware: it finishes filtering before CellShard emission, spills bounded build artifacts to a local spool, then assembles `dataset.csh5` and the active pack generation without rereading the expensive source.
- Single-machine operation should still follow the owner-service contract: `.csh5` stays under one owner-side coordinator and master reader, while local executors consume published pack generations.
- Distributed operation keeps `.csh5` on the owner host and delivers pack generations to executor nodes; remote nodes may cache pack data locally, but those caches are runtime artifacts rather than sources of truth.
- `docs/` now hosts the Cellerator pipeline manual plus local-only workflow notes; it is still not part of the default build or packaging surface.

## Python Wrapper

The root `cellerator` Python package is intentionally narrow. It wraps the dataset workbench library seam for:

- manifest and source inspection
- ingest planning and conversion to `.csh5`
- dataset summary and preprocess execution
- high-level reopening of the written dataset through `cellshard.open()`

It does not try to expose the full Torch/model/custom-kernel surface. For ordinary Python use:

```bash
python -m pip install extern/CellShard
python -m pip install .
```

```python
import cellerator

builder = cellerator.DatasetBuilder.from_manifest("manifest.tsv")
policy = cellerator.ingest_policy()
policy.output_path = "dataset.csh5"
plan = builder.plan(policy)
builder.convert(plan)
summary = builder.inspect_output()
dataset = builder.open()
```

## Where To Read Next

- `extern/CellShard/README.md`: CellShard scope and storage/runtime details
- `AGENTS.md`: contributor rules, codebase conventions, and repository-specific guidance
- `optimization.md`: performance notes and bottleneck analysis
- `pointer_migration_plan.md`: pointer-first migration policy for hot paths
- `todos.md`: active work ledger when substantial repo work is in progress
