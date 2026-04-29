# Cellerator

Cellerator is a GPU-oriented compute and model repository for large sparse single-cell datasets.

It sits on top of `CellShard`, which handles canonical sparse storage, optional ingest, finalize/repack staging, CSPACK delivery, and distributed execution staging. Cellerator owns math over CellShard matrices, including reusable sparse compute, model code, trajectory logic, quantized kernels, and Torch-facing boundaries. The biology-facing preprocessing API and workflow policy live in `extern/CellShardPreprocess`.

This codebase is built for explicit low-level control rather than a high-level workflow API. The durable scope contract is in `scope.md`: Cellerator owns sparse biological ML operators, training primitives, model components, distributed sparse execution, and explicit Torch interop. It does not replace Torch, AnnData, Scanpy, or CellShard, and the future compiled `.cellerator` model format is out of scope for now. If you want source ingest or storage workflows, see `extern/CellShard/`.

Surfaces that are temporarily inside this repo but do not belong in Cellerator long term are tracked in `out_of_scope_inventory.md`.

## What Is Here

- `src/compute/`: reusable sparse compute, neighbors, autograd, and custom-op building blocks
- `src/models/`: libtorch model workflows
- `src/trajectory/`: trajectory scoring and assembly
- `src/quantized/`: quantized sparse runtime
- `src/torch/`: explicit CellShard-to-Torch boundary
- `tests/`: compile and runtime checks
- `bench/`: benchmark binaries
- `extern/CellShard/`: storage, staging, and optional ingest submodule used by Cellerator
- `extern/CellShardPreprocess/`: biology-facing scRNA preprocessing policy, runtime API, and native workbench target

CellShardPreprocess exposes its native C++ API in `namespace cellshard_preprocess`.
Use `namespace cspre = ::cellshard_preprocess;` as the short Cellerator-facing
alias when calling that API from wrappers, tests, or examples.

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
- `src/compute/neighbors/`: nearest-neighbor math only; caller policy lives in `extern/CellShardNeighbors/`
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

CellShard ingest is optional and disabled by default. Build it from the CellShard subproject with `-DCELLSHARD_BUILD_INGEST=ON`; `-DCELLSHARD_INSTALL_PREPROCESS=ON` forces that flag on.

## Running Things

`ctest` is not configured. Run the built binaries directly.

Examples:

```bash
./build/cellShardDatasetH5Test
./build/forwardNeighborsCompileTest
./build/computeAutogradRuntimeTest
./build/quantizeModelTest
```

Useful build targets include:

- `cellShardDatasetH5Test`
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
- CellShard archive semantics are now assay-aware: `.csh5` and `.cshard` can describe a shared observation table plus per-assay sparse matrices, feature tables, row maps, and pairing metadata for exact or partial multiome lookup.
- CUDA mode selection is explicit: `generic` is the default topology-agnostic path, while `native` and `native-extreme` unlock the host-specific V100 ordering only after runtime discovery confirms that topology.
- Blocked-ELL is the native sparse execution and persistence layout for `.csh5` output; CSR/compressed remains an explicit interop/export path in memory, not a supported `.csh5` file format.
- `.cshard` is present as an experimental standby HDF5-free native archive: it can be inspected, validated, converted from `.csh5`, and read directly by sparse row range, but `.csh5` remains the production durable format.
- The Blocked-ELL tuner is storage-first for sparse omics ingest: it now minimizes padded scalar-value bytes before using block-occupancy fill as a tie-breaker, and the default candidate set includes `4x4` blocks so scRNA row-block unions do not automatically collapse into near-full-width `32x32` layouts.
- The native ingest path will try the CUDA COO-to-Blocked-ELL builder first, but it now validates the live nonzero count of each built part before publishing it. If the GPU builder produces a payload whose live entries do not match the source COO semantics, ingest falls back to the CPU builder for that part instead of writing a mismatched optimized shard.
- Optimized blocked `.csh5` files now persist only `/payload/optimized_blocked_ell` shard blobs plus matrix metadata; they no longer carry a second heavyweight `/payload/blocked_ell` body in the same file. Compatibility paths that still need canonical blocked partitions reconstruct them lazily from the optimized shard payload or from published CSPACK files.
- Optimized blocked shard blobs also use a compact remap encoding: shard-level column permutations are stored once, inverse permutations are reconstructed on load, and permutation arrays are narrowed to `u8`/`u16` on disk when the shard shape permits it.
- CellShard owns source ingest when configured with `CELLSHARD_BUILD_INGEST=ON`; Cellerator no longer exposes ingest or workbench compatibility APIs from the root build.
- Cellerator owns model-facing numerical math over CellShard matrices. CellShardPreprocess owns native QC rules, normalization/log1p, keep-mask policy, and raw-count validation as biology-facing workflow policy. CellShard owns data handling plus generic runtime sparse row/feature masking and grouped row reductions over its execution layouts.
- C++ call sites should spell the native preprocessing namespace as `cspre` via `namespace cspre = ::cellshard_preprocess;`; keep the full `cellshard_preprocess` spelling for declarations and public documentation that names the ABI directly.
- Single-machine operation should still follow the owner-service contract: `.csh5` stays under one owner-side coordinator and master reader, while local executors consume published CSPACK generations.
- Distributed operation keeps `.csh5` on the owner host and delivers CSPACK generations to executor nodes; remote nodes may cache CSPACK data locally, but those caches are runtime artifacts rather than sources of truth.
- CSPACK remains a single-assay execution matrix artifact. Multiome execution uses coordinated per-assay CSPACK files plus manifest metadata that records `assay_id`, global observation ranges, local row ranges, generation, and layout.
- User-defined annotations now live in cold `/observation_metadata`, `/feature_metadata`, and `/dataset_attributes` groups. Owner snapshots advertise those surfaces, but they are fetched only on explicit request rather than being shipped on the hot CSPACK bootstrap path.
- `docs/` now hosts the Cellerator pipeline manual plus local-only workflow notes; it is still not part of the default build or packaging surface.
- `docs/quantized_transfer_architecture.qmd` records the current Volta-oriented report for quantized sparse transport, decode cost, and tensor-op-oriented follow-up work.

## Where To Read Next

- `extern/CellShard/README.md`: CellShard scope and storage/runtime details
- `extern/CellShardPreprocess/README.md`: CellShardPreprocess preprocessing runtime scope
- `scope.md`: canonical Cellerator ownership and out-of-scope boundary
- `out_of_scope_inventory.md`: advisory queue for code that should move out or be reclassified
- `planning_strategy.md`: pickup guide for operator-first sparse biological ML planning
- `AGENTS.md`: contributor rules, codebase conventions, and repository-specific guidance
- `optimization.md`: performance notes and bottleneck analysis
- `pointer_migration_plan.md`: pointer-first migration policy for hot paths
- `todos.md`: active work ledger when substantial repo work is in progress
