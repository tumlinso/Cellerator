# Cellerator

Cellerator is a GPU-oriented compute and model repository for large sparse single-cell datasets.

It sits on top of `CellShard`, which handles canonical sparse storage, finalize/repack staging, pack delivery, and distributed execution staging. Cellerator adds ingest, one-pass preprocessing/filtering, reusable sparse compute, model code, trajectory logic, Torch-facing boundaries, and the ncurses workbench.

This codebase is built for explicit low-level control rather than a high-level workflow API, but it now also carries a thin Python wrapper for the dataset-facing workbench path. If you want the storage layer by itself, see `extern/CellShard/`.

## What Is Here

- `src/ingest/`: source readers, MTX conversion, and dataset ingest
- `src/compute/`: reusable sparse compute, preprocess operators, neighbors, and custom-op building blocks
- `src/models/`: libtorch model workflows
- `src/trajectory/`: trajectory scoring and assembly
- `src/workbench/`: keyboard-driven terminal workbench adapters and shared runtime facade
- `src/apps/`: application entrypoints such as the ncurses workbench binary
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

- `include/Cellerator/`: canonical public headers for in-repo callers
- `src/compute/`: low-level compute building blocks and performance-sensitive kernels
- `src/compute/autograd/`: sparse autograd contexts, scratch, and kernels
- `src/compute/neighbors/`: nearest-neighbor and retrieval code
- `src/ingest/`: source-format readers, metadata tables, partition planning, and dataset conversion
- `src/models/`: header-first model modules, typically split into `*_dataloader.hh`, `*_model.hh`, `*_train.hh`, and `*_infer.hh`
- `src/workbench/`: the ncurses workbench application, public facade, and `runtime/` backend
- `src/apps/`: adapter or executable entrypoints
- `src/quantized/`: quantized sparse layouts, accessors, and kernels
- `src/torch/`: explicit Torch interop boundary

Public in-repo includes should use `#include <Cellerator/...>` rather than reaching into `src/` directly.

## Quick Start

Configure and build:

```bash
cmake -S . -B build
cmake --build build -j 4
```

CUDA mode defaults to `generic`. Use `-DCELLERATOR_CUDA_MODE=native` for the
host-specific fast path and `-DCELLERATOR_CUDA_MODE=native-extreme` for the
separate Volta-only/PTX-heavy path.

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
./build/cellShardFirstFileFixtureTest
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
- `cellShardFirstFileFixtureTest`
- `datasetIngestCompileTest`
- `datasetWorkbenchRuntimeTest`
- `forwardNeighborsCompileTest`
- `computeAutogradRuntimeTest`
- `quantizedTransferSpmmBench`
- `developmentalTimeCompileTest`
- `denseReduceCompileTest`
- `quantizeModelTest`
- `torchBindingsCompileTest`
- `modelCustomOpsTest`

## Repository Notes

- Cellerator is performance-oriented and currently tuned around Volta / V100-class assumptions.
- CUDA mode selection is explicit: `generic` is the default topology-agnostic path, while `native` and `native-extreme` unlock the host-specific V100 ordering only after runtime discovery confirms that topology.
- Blocked-ELL is the native sparse execution and persistence layout for `.csh5` output; CSR/compressed remains an explicit interop/export path in memory, not a supported `.csh5` file format.
- The Blocked-ELL tuner is storage-first for sparse omics ingest: it now minimizes padded scalar-value bytes before using block-occupancy fill as a tie-breaker, and the default candidate set includes `4x4` blocks so scRNA row-block unions do not automatically collapse into near-full-width `32x32` layouts.
- The native ingest path will try the CUDA COO-to-Blocked-ELL builder first, but it now validates the live nonzero count of each built part before publishing it. If the GPU builder produces a payload whose live entries do not match the source COO semantics, ingest falls back to the CPU builder for that part instead of writing a mismatched optimized shard.
- Optimized blocked `.csh5` files now persist only `/payload/optimized_blocked_ell` shard blobs plus matrix metadata; they no longer carry a second heavyweight `/payload/blocked_ell` body in the same file. Compatibility paths that still need canonical blocked partitions reconstruct them lazily from the optimized shard payload or from published packs.
- Optimized blocked shard blobs also use a compact remap encoding: shard-level column permutations are stored once, inverse permutations are reconstructed on load, and permutation arrays are narrowed to `u8`/`u16` on disk when the shard shape permits it.
- Current MTX-dataset ingest is bounded-memory and SSD-aware: it emits the canonical `dataset.csh5`, spills bounded `.cspool` build artifacts to a local spool, and leaves shape-changing preprocess compaction to the later workbench preprocess/finalize path.
- Workbench preprocess now defaults to a two-stage flow: Cellerator computes QC, normalization/log1p metrics, keep masks, and metadata updates first, then CellShard finalizes the compacted Blocked-ELL dataset in place and rebuilds browse metadata. Set `preprocess_config.finalize_after_preprocess = false` only when you intentionally want a metadata-only benchmark stop before repack.
- Single-machine operation should still follow the owner-service contract: `.csh5` stays under one owner-side coordinator and master reader, while local executors consume published pack generations.
- Distributed operation keeps `.csh5` on the owner host and delivers pack generations to executor nodes; remote nodes may cache pack data locally, but those caches are runtime artifacts rather than sources of truth.
- User-defined annotations now live in cold `/observation_metadata`, `/feature_metadata`, and `/dataset_attributes` groups. Owner snapshots advertise those surfaces, but they are fetched only on explicit request rather than being shipped on the hot pack/bootstrap path.
- `docs/` now hosts the Cellerator pipeline manual plus local-only workflow notes; it is still not part of the default build or packaging surface.
- `docs/quantized_transfer_architecture.qmd` records the current Volta-oriented report for quantized sparse transport, decode cost, and tensor-op-oriented follow-up work.

## Python Wrapper

The root `cellerator` Python package is intentionally narrow. It wraps the dataset workbench library seam for:

- manifest and source inspection
- ingest planning and conversion to `.csh5`
- dataset summary and preprocess execution
- high-level reopening of the written dataset through `cellshard.open()`
- explicit cold metadata fetches for observation annotations, feature annotations, and dataset attributes through the CellShard owner/client handles

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
