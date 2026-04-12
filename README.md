# Cellerator

Cellerator is the active single-cell compute and modeling stack in this repository.

It sits on top of `CellShard`, which owns storage, packfile layout, fetch/drop, and staging substrate. Cellerator owns the computation that acts on that data: ingest orchestration, sparse preprocessing, neighbor and trajectory operators, model code, and the quantized runtime surfaces that support those workflows.

This is a low-level, performance-first codebase. The goal is not to hide the runtime behind abstracted-away building blocks. The goal is to expose the real hot-path choices clearly enough that layout, residency, transfer, launch, and kernel decisions can be made deliberately and optimized for the actual V100 target.

## Scope

`Cellerator` owns:

- source-format ingest and series conversion built on `CellShard`
- sparse preprocessing and normalization operators
- reusable compute kernels, workspaces, and multi-GPU compute surfaces
- trajectory and forward-neighbor search operators
- libtorch-facing model code
- the explicit Torch interop boundary
- low-level quantized matrix backends used by the model stack

`CellShard` owns:

- durable on-disk matrix layouts
- sharded storage metadata
- packfile and container layout
- fetch/drop and staging primitives
- low-level persisted sparse formats

Short version:

- `CellShard` stores and stages data
- `Cellerator` computes on it

## Current Layout

The active tree is no longer just ingest-heavy. Current non-scaffold homes include:

- `src/apps/`: ncurses workbench and command-line entrypoints
- `src/compute/`: authoritative home for reusable computation, including preprocess, neighbors, model ops, and the pointer-first sparse autograd building-block layer
- `src/ingest/`: source-format readers plus MTX and series ingest/orchestration
- `src/quantized/`: CUDA-first quantized CSR backend
- `src/models/`: header-first libtorch workflows such as `developmental_time`, `dense_reduce`, and `quantize`
- `src/support/`: small shared support headers used across active compute modules
- `src/torch/`: explicit, intentionally expensive Torch export boundary
- `src/trajectory/`: trajectory record, candidate, prune, and tree-building logic
- `src/legacy/`: quarantine area for old monolithic entrypoints while reusable code is extracted outward

## Directory Roles

### `src/compute/`

This is the main reusable compute surface in Cellerator.

It owns:

- sparse preprocess operators and workspaces
- exact and approximate neighbor search operators
- forward-neighbor index and query operators
- framework-independent sparse autograd/runtime code
- Torch custom CUDA model ops

This directory should stay concrete. Reusable kernels and operators belong here; one-off orchestration and abstraction-heavy wrappers do not.

`src/compute/autograd/` is a particularly important example of that rule. It is not a generic framework surface. It is a pointer-first sparse building-block library with:

- explicit CSR-first APIs
- FP16 storage with FP32 accumulate on the main path
- single-GPU `base::` kernels
- distributed `dist::launch_*` entrypoints over selected device slots
- explicit pair-local and leader-merge reductions for the real 4-GPU topology

That layer exists so sparse model math can stay close to the actual layout, stream, and device-topology costs instead of being abstracted upward behind a heavier runtime.

### `src/apps/`

Interactive and command-line frontends live here.

Current code includes the ncurses series workbench plus its supporting ingest inspection and preprocess orchestration helpers.

### `src/ingest/`

This is the active source-format ingest home.

It covers:

- MTX parsing and partition planning
- compressed-part conversion workspace
- manifest-driven series ingest and HDF5 emission
- shared ingest tables for features, barcodes, and metadata

This layer should remain explicit about CPU parsing, pinned staging, transfer boundaries, and file-format constraints.

### `src/models/`

This is the active libtorch model surface.

Current modules include:

- `developmental_time/`
- `dense_reduce/`
- `quantize/`

Each workflow follows the header-first split already in the tree:

- `*_dataloader.hh`
- `*_model.hh`
- `*_train.hh`
- `*_infer.hh`
- one umbrella header per workflow

Forward-neighbor retrieval is not a `src/models/` module. It lives under `src/compute/neighbors/forward_neighbors/` because it is a reusable compute operator rather than a model wrapper.

### `src/trajectory/`

Trajectory code currently combines GPU candidate scoring with CPU-side graph assembly and pruning. It is an active subsystem, not a placeholder.

### `src/torch/`

This is a narrow, explicit interoperability boundary. It should not become the core execution model of the repo. Export here is intentionally copy-based and expensive so ownership and performance costs stay visible.

### `src/quantized/`

This is the low-level quantized backend. It is intentionally CUDA-first and layout-explicit. It should not be replaced with a more generic abstraction just to make the code look cleaner at the cost of hot-path control.

Near-term direction: this quantized path is the first intended consumer of a hand-rolled gradient calculator. If custom gradient work starts somewhere in the repo, it should start here rather than by wrapping the quantized path in a more abstract training surface.

Public includes should come from `src/quantized/api.cuh` rather than the removed legacy camel-case umbrellas.

## Design Direction

Cellerator is trying to be a serious sparse runtime and modeling engine, not a generic high-level framework.

That means:

- low-level optimization is a feature, not technical debt
- hot paths should stay honest about memory layout, allocation, transfers, and synchronization
- reusable building blocks should remain concrete enough that they can still be profiled and tuned
- “abstracting away” kernels, residency, or sparse layout boundaries is usually the wrong direction for this repo

If a simpler abstraction blocks the fastest reasonable Volta path, the abstraction should lose.

## Build

Configure with:

```bash
cmake -S . -B build
```

Build with:

```bash
cmake --build build -j 4
```

Useful current targets include:

- `scrnaPreprocessBench`
- `cellShardSeriesH5Test`
- `quantizedMatrixTest`
- `seriesIngestCompileTest`
- `seriesWorkbenchRuntimeTest`
- `celleratorWorkbench`
- `trajectoryCompileTest`
- `trajectoryRuntimeTest`
- `forwardNeighborsCompileTest`
- `computeAutogradRuntimeTest`
- `developmentalTimeCompileTest`
- `denseReduceCompileTest`
- `quantizeModelTest`
- `torchBindingsCompileTest`
- `modelCustomOpsTest`

Legacy `matrixTest` and `convertEmbryoMtxShards` targets are only built when `src/matrix/matrix_io.cu` exists in the checkout. Their monolithic source now lives under `src/legacy/monoliths/`.

## Torch And Toolchain Notes

Torch-enabled builds prefer the source-built libtorch installation under `/usr/local/share/cmake/Torch`. If libtorch is unavailable, configure with:

```bash
cmake -S . -B build -DCELLERATOR_ENABLE_TORCH_MODELS=OFF
```

Only override the default libtorch with `Torch_DIR` or `LIBTORCH_PATH` when you intentionally want a different build.

CMake defaults to the local HPC SDK CUDA toolchain and `g++-12` host compiler when no override is supplied. Use `CUDACXX` and `CUDAHOSTCXX` for explicit overrides.

## Testing

`ctest` is not configured. Run built targets directly, for example:

```bash
./build/cellShardSeriesH5Test
./build/forwardNeighborsCompileTest
./build/computeAutogradRuntimeTest
./build/quantizeModelTest
```

When touching a GPU-facing path, prefer reporting the exact build and run commands used rather than relying on a generic “tests passed” summary.

## Optimization Posture

The repository target is Volta `sm_70` on Tesla V100 16 GB GPUs.

Optimization priorities here are mostly:

- keep sparse data in one useful layout
- keep it resident on device when possible
- preallocate and reuse workspaces
- avoid needless host round-trips
- avoid format churn across model and compute boundaries

Do not assume the right answer is to wrap low-level pieces in more generic infrastructure. In this repo, the building blocks are supposed to stay visible enough to optimize.
