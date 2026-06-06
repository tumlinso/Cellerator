# Cellerator

Cellerator is a GPU-oriented preprocessing, math, compute, and model package for large sparse single-cell datasets in the CellStack family.

It consumes CellShard as an external dependency for sparse matrix/runtime ABI types. Cellerator owns preprocessing and math over those matrices: reusable sparse compute, forward-neighbor index/query policy, model code, trajectory logic, quantized kernels, and Torch-facing boundaries. Storage and ingest live in CellShard.

This codebase is built for explicit low-level control rather than a high-level workflow API. The durable scope contract is in `scope.md`: Cellerator owns sparse biological ML operators, training primitives, model components, distributed sparse execution, and explicit Torch interop. It does not replace Torch, AnnData, Scanpy, or CellShard, and the future compiled `.cellerator` model format is out of scope for now.

Surfaces that are temporarily inside this repo but do not belong in Cellerator long term are tracked in `out_of_scope_inventory.md`.

## What Is Here

- `src/compute/`: reusable sparse compute, neighbors, sparse operators, and custom-op building blocks
- `src/preprocess/`: biology-facing preprocessing policy, runtime API, and workbench orchestration
- `src/models/`: libtorch model workflows
- `src/trajectory/`: trajectory scoring and assembly
- `include/Cellerator/core/`: CelleratorCore format, conversion, quantized packing, and CUDA substrate
- `src/torch/`: explicit CellShard-to-Torch boundary
- `tests/`: compile and runtime checks
- `bench/`: benchmark binaries

Expected local source checkout layout:

```text
cellstack/
├── Baseplane/
├── CellShard/
└── Cellerator/
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
- `include/Cellerator/core/matrix/`: CelleratorCore sparse layout ABI and device views
- `include/Cellerator/compute/matrix/convert/`: layout conversion, transpose, and bucket planning contracts
- `include/Cellerator/core/runtime/`: CellShard-free CUDA buffers, streams, scratch, and library handles
- `include/Cellerator/core/quantized/`: quantized format metadata, access, packing, and dispatch helpers
- `src/compute/`: sparse ML math over CellShard matrices
- `src/compute/sparse/ops/`: sparse operator contexts, scratch, and kernels
- `src/compute/neighbors/`: exact-search/scoring math plus forward-neighbor index/query policy
- `src/models/`: header-first model modules, typically split into `*_dataloader.hh`, `*_model.hh`, `*_train.hh`, and `*_infer.hh`
- `src/torch/`: explicit Torch interop boundary

Public in-repo includes should use `#include <Cellerator/...>` rather than reaching into `src/` directly.
CelleratorCore's `real` traits define `f16_t`, `f32_t`, `f64_t`, and, when the
CUDA toolchain exposes them, `bf16_t`, `fp8_e4m3_t`, and `fp8_e5m2_t`. Existing
sparse layouts keep `__half` as their default ABI storage type unless a caller
opts into a specific precision policy.

## Quick Start

Configure and build:

```bash
cmake -S . -B build
cmake --build build -j 4
```

Python package basics:

```bash
python -m pip install .
python -m pip wheel . --no-deps
```

The Python package is intentionally an orchestration layer, not a CPU
preprocessing implementation. `cellerator.pp.preprocess(...)` accepts a
CellShard `.csh5` path, a `cellshard.Dataset`, or an AnnData object that
explicitly carries `uns["cellshard_path"]`; the hot path stages CellShard-native
Blocked-ELL or Sliced-ELL partitions to GPU and delegates QC,
normalize-total/log1p, and metrics to Cellerator kernels. AnnData and SciPy are
adapter/egress surfaces only and are not used for hidden matrix transforms.

The returned `PreprocessSession` keeps preprocessing metrics and keep masks as
the explicit boundary. `session.publish(path)` delegates persistence to
CellShard's preprocessed dataset finalize path, which publishes filtered
CellShard data plus preprocessing metadata while storage ownership remains in
CellShard.

Python preprocessing can optionally run a bounded light autotune pass with
`cellerator.pp.preprocess(path, autotune=True)`. The first provider samples an
eligible small CellShard partition, compares the fused preprocessing traversal
against the separate primitive sequence, and keeps the default fused path when
the measured difference is within the close-enough threshold or when sampling
would cost too much for the dataset. C++ mode remains explicit: callers can
benchmark and choose the plan-aware preprocessing primitives directly, while
fleet session preprocessing keeps the fused default unless that caller wires its
own optimizer policy around those primitives.

CMake resolves CellShard in this order:

1. `-DCELLERATOR_CELLSHARD_SOURCE_DIR=/path/to/CellShard`
2. sibling `../CellShard`
3. `find_package(CellShard CONFIG REQUIRED)`

CMake resolves Baseplane sequence bit primitives in this order:

1. `-DBASEPLANE_SOURCE_DIR=/path/to/Baseplane`
2. sibling `../Baseplane`
3. `find_package(Baseplane CONFIG REQUIRED)`

CUDA mode defaults to `generic`. Use `-DCELLERATOR_CUDA_MODE=native` for the
host-specific fast path and `-DCELLERATOR_CUDA_MODE=native-extreme` for the
separate Volta-only/PTX-heavy path.

The generated CelleratorCore config also records default scalar and policy
choices for new, unspecified work:

```bash
cmake -S . -B build \
  -DCELLERATOR_REAL_STORAGE=f16 \
  -DCELLERATOR_REAL_COMPUTE=f32 \
  -DCELLERATOR_REAL_ACCUM=f32 \
  -DCELLERATOR_DEFAULT_SPARSE_LAYOUT=blocked_ell \
  -DCELLERATOR_DEFAULT_TRAINING_PRECISION=f32 \
  -DCELLERATOR_DEFAULT_GRADIENT_CLIPPING=none
```

Configure also probes the local host by default:

```bash
cmake -S . -B build \
  -DCELLERATOR_ENABLE_HARDWARE_PROBE=ON \
  -DCELLERATOR_AUTO_DETECT_CUDA_ARCHITECTURES=ON
```

The generated config exposes CPU vector facts such as AVX2/AVX-512/F16C/FMA
and CUDA facts such as visible device count, selected device, compute
capability, SM architecture, memory, warp size, and tensor-core family support.
This hardware information is advisory; explicit caller/file metadata still wins
when loading stored payloads or selecting a precision-specific execution path.

Local and generated fixture payloads belong under `data/`. The scaffold is
tracked, while large biological data files and generated conversion outputs are
ignored by default.

These are build defaults, not file-format assumptions. CellShard `.csh5` and
`.cspack` payloads remain self-describing through codec/layout metadata such as
`value_code` and `bits`; readers must dispatch or convert from the stored
precision instead of reinterpreting payloads as the local build default.

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
./build/sparseOpsRuntimeTest
./build/quantizeModelTest
```

Useful build targets include:

- `quantizedMatrixTest`
- `trajectoryCompileTest`
- `trajectoryRuntimeTest`
- `exactSearchRuntimeTest`
- `sparseOpsRuntimeTest`
- `developmentalTimeCompileTest`
- `denseReduceCompileTest`
- `quantizeModelTest`
- `torchBindingsCompileTest`
- `modelCustomOpsTest`

## Repository Notes

- Cellerator is performance-oriented and currently tuned around Volta / V100-class assumptions.
- CUDA mode selection is explicit: `generic` is the default topology-agnostic path, while `native` and `native-extreme` unlock the host-specific V100 ordering only after runtime discovery confirms that topology.
- Blocked-ELL is the preferred native sparse execution layout for CelleratorCore/Cellerator hot paths; CSR/compressed remains an explicit fallback or interop representation where a surface still requires it.
- CelleratorCore owns substrate and format mechanics; `src/compute/` owns math such as sparse projection, matmul, ML reductions, and training operators.
- Cellerator owns preprocessing, model-facing numerical math over CellShard matrices, and the forward-neighbor caller surface built on that math. Data handling and source ingest stay in CellShard.
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
