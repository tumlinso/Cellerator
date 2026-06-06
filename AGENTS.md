# Repository Guidelines

## Project Structure & Module Organization
`include/Cellerator/` is the canonical public include tree for in-repo callers. `src/` holds compiled implementation code for Core and higher Cellerator layers. Primary non-Core surfaces live under `src/compute/`, `src/preprocess/`, `src/models/`, and `src/trajectory/`. Torch-facing integration lives under `components/CelleraTorch/`. `tests/` contains native compile and runtime checks, and `bench/` contains performance benches. Cellerator is a standalone CellStack math/compute and preprocessing package; storage and durable dataset publication live in CellShard. Forward-neighbor caller policy is currently under `src/compute/neighbors/forward_neighbors/` while the sister-project split is still in progress.

The durable Cellerator scope boundary lives in `scope.md`, and the advisory migration queue for surfaces that do not belong in Cellerator lives in `out_of_scope_inventory.md`. Read both before adding public APIs, new CMake targets, model modules, preprocessing code, ingest/runtime code, or Torch boundaries. If work touches scope drift, remind the user that `out_of_scope_inventory.md` is the migration queue and update it before normalizing the drift as Cellerator-owned.

Current direction: finish Cellerator as the CellStack base math and execution
engine before spending attention on new human-facing UI/API surfaces. CellShard
is the only project with the beginnings of a publishable user API, but a
meaningful CellShard release depends on Cellerator being complete enough to
provide the sparse math, preprocessing engines, neighbor/search engines, layout
optimization, and file/runtime build primitives that CellShard needs. Treat
Cellerator completion as the current priority.

Preprocessing and neighbor/search work are not automatically out of scope for
Cellerator. They may remain native Cellerator capabilities when they are
implemented as layout-aware engines, metric/stat kernels, optimization inputs,
or execution primitives used to build and optimize Blocked-ELL/Sliced-ELL files
and runtime payloads. Do not spend current effort designing Scanpy-like
preprocessing APIs, neighbor workflow ergonomics, UI/workbench behavior, or
other human-facing product surfaces unless explicitly requested. If those
surfaces are exposed later, they should be thin wrappers over the native engines
instead of a separate math implementation.

Torch/libtorch is the clear split exception: do not expand Torch-linked code as
core Cellerator. Its intended destination is the separate `CelleraTorch` project
staged under `components/CelleraTorch/`. Keep reusable kernels, layouts,
reductions, sparse transforms, quantized primitives, and runtime scratch
mechanics in native Cellerator, and keep framework adapters in CelleraTorch.

`include/Cellerator/core/` is the CelleratorCore public surface for matrix representation ABI, parameter descriptors, quantized packing/metadata, CellShard-free runtime substrate, and interop contracts. Sequence bit primitives live in the sibling Baseplane project, not in Cellerator. Its contract-first layout is `core/matrix/`, `core/runtime/`, `core/quantized/`, and `core/interop/`. `src/core/` contains the compiled Core implementation behind the single `Cellerator::core` target. `src/compute/` is the authoritative home for reusable math and operators: matrix conversion and bucketing, CUDA compute primitives, exact search math, preprocessing math kernels, sparse projection/matmul, ML reductions, and the sparse operator layer under `src/compute/sparse/ops/`. `components/CelleraTorch/` owns Torch tensor export, Torch custom-op wrappers, Torch-linked quantizer wrappers, and the legacy dense-reduce prototype. `src/preprocess/` owns biology-facing preprocessing policy, raw-count state validation, QC rule compilation, adapter staging, and workbench orchestration over `Cellerator::compute_preprocess`. `src/compute/runtime/runtime.hh` keeps CellShard-aware fleet execution and reuses the Core runtime substrate. Native `src/models/` contains Cellerator-owned model implementations that do not expose framework tensor types. Neighbor retrieval index/query policy is not a Cellerator public API.

Workflow code must not own reusable math. If preprocessing, model, Torch,
neighbor, or session/workbench code needs GPU math, reductions, dense adds,
metric packets, sparse transforms, normalization, fleet collectives, or scratch
mechanics, call the existing `src/compute/` primitive. If the primitive does
not exist, implement the primitive in the owning compute layer first and make
the workflow call it. Do not copy kernels or math into workflow/session files
for convenience.

Implementation organization follows `style_hint.md`. Read it before changing
preprocessing compute/runtime code, adding private CUDA helpers, or extending a
large CUDA implementation file. It is the local rule for file-size pressure,
inline device helpers, fused workflow placement, and the current monolith
follow-up list.

Expected local checkout layout for source builds is:

```text
cellstack/
├── CellShard/
└── Cellerator/
```

For substantial repo work, consult root `todos.md` first. Detailed workstream ledgers under `todos/` are optional and may not exist.

For custom gradients, the first intended handwritten gradient-calculator target is the quantized path. Treat that backend as the leading candidate for explicit backward logic before introducing broader custom-gradient machinery elsewhere.

## Build, Test, and Development Commands
Configure with `cmake -S . -B build` and build with `cmake --build build -j 4`. CMake resolves CellShard in this order: `CELLERATOR_CELLSHARD_SOURCE_DIR`, sibling `../CellShard`, then `find_package(CellShard CONFIG REQUIRED)`. Useful native Cellerator targets include `quantizedMatrixTest`, `trajectoryCompileTest`, `trajectoryRuntimeTest`, `exactSearchRuntimeTest`, `sparseOpsRuntimeTest`, and `developmentalTimeCompileTest`. CelleraTorch targets include `celleraTorchBindingsCompileTest`, `celleraTorchDenseReduceCompileTest`, `celleraTorchQuantizePrimitiveTest`, and `celleraTorchModelCustomOpsTest`. Run built tests directly because `ctest` is not configured, for example `./build/exactSearchRuntimeTest`, `./build/sparseOpsRuntimeTest`, or `./build/celleraTorchQuantizePrimitiveTest`. If CelleraTorch dependencies are unavailable, configure with `cmake -S . -B build -DCELLERATOR_ENABLE_TORCH_MODELS=OFF`; use `Torch_DIR` or `LIBTORCH_PATH` only when you intentionally need to override the component default.

## Coding Style & Naming Conventions
Follow the existing C++17/CUDA17 style: 4-space indentation, opening braces on the same line, and standard-library names qualified with `std::` when they are used. Prefer `snake_case` for functions, variables, structs, and CLI flags; use short type aliases only when they improve readability. Match current file suffixes: `.cu` for CUDA translation units, `.cuh` for CUDA headers, `.cc`/`.hh` for C++ sources and headers. Keep file/storage orchestration and ingest outside Cellerator; keep preprocessing, sparse layout primitives, math, forward-neighbor orchestration, and model logic in CelleratorCore/Cellerator.

The old preprocessing project names have been retired. If future work encounters `MosaiCell`, `Mosaicell`, `mosaicell`, `mosaiCell`, or `CellShardPreprocess`, change it automatically to Cellerator preprocessing names unless the text is explicitly historical. For C++ caller aliases, prefer `cpre`.

Bias new code toward HPC-first implementation choices. Performance on Volta `sm_70` takes priority over genericity, portability, or abstraction count in performance-sensitive code. Do not keep a standard-library abstraction in a hot path just because it is idiomatic C++ if a lower-level representation is measurably or obviously better for throughput, copy behavior, launch behavior, memory traffic, or layout control.

This repository is not trying to abstract low-level runtime building blocks away. Kernels, layouts, residency boundaries, scratch buffers, and launch structure are supposed to remain visible enough to optimize directly. If a “cleaner” abstraction hides the real cost model or blocks the best Volta path, it is the wrong fit for this codebase.

The repository-wide migration plan for removing vector-heavy hot-path code lives in `pointer_migration_plan.md` at the repo root. Read that file before changing any of the current heavy-violation subsystems or introducing a new GPU-facing buffer surface.

`std::vector` is not the default container for GPU-facing or copy-sensitive code, and it is not an acceptable steady-state container in hot paths. Prefer explicit contiguous buffers, pointer-plus-size interfaces, fixed-capacity or preallocated storage, trivially copyable structs, and layout choices that make host-device transfers and kernel access patterns obvious. Use raw pointers, `std::unique_ptr<T[]>`, aligned allocations, or library-owned buffers when they reduce hidden reallocations, iterator-heavy code, ownership ambiguity, or extra copies. Keep `std::vector`, `std::string`, `std::function`, stream-heavy I/O, and similar abstractions out of kernels, transfer boundaries, repeated batch assembly, merge scratch, and tight preprocessing loops unless they are clearly off the hot path or have been shown not to matter.

Do not introduce new `std::vector`-based public APIs for hot subsystems. When touching an existing vector-heavy hot path, bias the change toward the pointer-first end state described in `pointer_migration_plan.md` rather than preserving the current container surface.

Prefer structure-of-arrays over array-of-structures when access is columnar or warp-coalescing benefits. Preallocate aggressively, fuse passes when that removes HBM traffic, avoid allocator churn inside repeated workflows, and keep data resident on device whenever feasible. If a simpler but slower abstraction is retained, document the reason or the measurement that justified it.

When several local variables share the same type and form one obvious working set, prefer compressed declarations on one statement instead of one-per-line boilerplate so readers do not need to scan extra vertical context just to recover the type; split them back out only when initialization, ownership, comments, or future type drift make that clearer.

For native `src/models/`, keep framework-linked wrappers out of Cellerator. New native model workflows should keep learned parameters and execution buffers visible enough for direct CUDA optimization and future `core/parameters.hh` descriptor exposure.

## Testing Guidelines
Add tests under `tests/` beside the nearest feature area and name them after the unit or workflow being checked, for example `series_ingest_compile_test.cu`. Cover both compile-only integration points and small runtime checks when behavior can be exercised locally. For model work, prefer compile coverage for the umbrella header and a focused runtime test for loss, inference, or retrieval behavior when the path can run locally. For GPU-facing changes, build the affected target and run the corresponding binary; include the exact command in your PR notes.

Benchmark and profiler runs must be serialized across workers. Treat the repository-wide benchmark mutex as mandatory so concurrent agents do not skew GPU measurements. When adding or changing a benchmark binary under `bench/`, wire it through `bench/benchmark_mutex.hh` and do not bypass that lock in normal benchmark workflows.

Model-facing and compute-adjacent targets currently map to:
- `developmentalTimeCompileTest` for `src/models/developmental_time/`
- `exactSearchRuntimeTest` for `src/compute/neighbors/exact_search/`
- `sparseOpsRuntimeTest` for `src/compute/sparse/ops/`
- `celleraTorchBindingsCompileTest` for `components/CelleraTorch/` tensor export
- `celleraTorchDenseReduceCompileTest` for the CelleraTorch dense-reduce prototype
- `celleraTorchQuantizePrimitiveTest` for CelleraTorch quantizer wrappers
- `celleraTorchModelCustomOpsTest` for CelleraTorch custom-op wrappers

## Model Design And Skills
When the task is deciding what model family, objective, latent structure, decoder, or loss should exist in `src/models/`, use `$v100-model-design` first. That includes new developmental-time models, latent reduction variants, multimodal or temporal extensions, and decisions about whether the work should stay in libtorch or begin in Python PyTorch with a later C++ path. Stay in that skill until the model choice, scaling posture, and any custom-op boundary are stable.

When the task moves from model choice to implementation constraints on the 4x V100 host, use `$cuda-v100` as the default path. That includes `sm_70` build assumptions, CUDA 12.x compatibility, memory fit, host-device staging, NCCL or DDP topology, quantized kernels, profiler-driven tuning, and any custom Torch CUDA extension boundary. Follow the skill's path-selection logic instead of treating CUDA work as generic GPU programming: choose the matching V100 path first, load only the relevant addendum or base reference, and optimize for the real bottleneck.

For repository-specific performance context, read `optimization.md` at the repo root before making or defending changes in hot paths. It documents the current subsystem-level bottlenecks, likely fixed-call overheads, V100-oriented optimization priorities, and the repo's bias toward explicit low-level building blocks over abstraction-heavy surfaces.

Sparse layout policy in CelleratorCore remains Blocked-ELL-first. Treat Blocked-ELL as the native sparse type for persisted Cellerator execution, staging, and hot-path compute unless a subsystem is explicitly operating in a fallback compressed-only mode. Cellerator preprocessing treats Blocked-ELL and Sliced-ELL as first-class preprocessing layouts, with compressed / CSR as fallback. Do not describe compressed / CSR as the default sparse representation in new Cellerator code or docs unless that specific surface really is still fallback-only.

For migration work aimed at removing `std::vector` and abstraction from hot code, also read `pointer_migration_plan.md` at the repo root. It defines the subsystem ordering, target representations, and exit criteria for the pointer-first rewrite.

For manuscript planning, model framing, or conceptual discussion about regulatory dynamics, `docs/notes.txt` contains background notes from an unfinished separate project that may still be useful as reference context. Treat `docs/notes.txt` as optional background material rather than current manuscript source or repository ground truth unless the user explicitly asks to build from it.

If work around `docs/`, notes, or manuscript-like material turns into writing, figure generation, or citation support, use `$quarto-manuscript` rather than splitting those tasks across separate Quarto skills.

When code changes materially alter runtime, storage, ingest, pack, or other pipeline behavior, update the corresponding documentation in `docs/` and any primary README surface that describes that behavior as part of the same change. Do not leave behavior documentation stale after the implementation lands.

If new model work appears to need framework custom ops, place that adapter under `components/CelleraTorch/` and keep the op scope minimal. Prefer library-backed framework, cuBLAS, cuSPARSE, or CUTLASS paths before adding handwritten CUDA.

For CUDA/C++ implementation work outside `src/models/`, default to `$cuda-v100` whenever kernel shape, memory fit, communication topology, HtoD staging, sparse layout, or profiler interpretation is material to the answer. Recommendations should explicitly say whether they are library-backed or custom-kernel, whether they assume `sm_70`, and what the dominant limiter is: PCIe, HBM traffic, occupancy, register pressure, launch overhead, or cross-GPU communication.

Do not preserve a higher-level abstraction purely for compatibility if it blocks the fastest reasonable Volta path. Prefer explicit data layouts, pointer-stable ownership, fused kernels, and pair-local communication patterns when they are the better performance choice.

## Commit & Pull Request Guidelines
Recent commits use short, lowercase summaries such as `move ingest to smellerator` and `reduce & time models`. Keep commit subjects brief, specific, and scoped to one change. PRs should explain the affected module, list build/test commands run, note any CUDA/HDF5/libtorch assumptions, and include benchmark deltas when touching kernels, ingest throughput, or preprocessing performance.

## Configuration Notes
CMake defaults to the local HPC SDK CUDA toolchain and `g++-12` host compiler when no override is supplied. Prefer explicit overrides with `CUDACXX` and `CUDAHOSTCXX` if you need a different toolchain.

For CelleraTorch, prefer the source-built installation under `/usr/local` over a Python-packaged or PyTorch-bundled build. That `/usr/local` build is the repository default because it is tuned for this host's V100/NVLink layout. Only point CMake at another dependency root with `Torch_DIR` or `LIBTORCH_PATH` when you are deliberately testing or debugging against a different build.

Model and quantization work should assume the repository is targeting Volta `sm_70` on Tesla V100 16 GB GPUs unless the task says otherwise. Treat PCIe as a bottleneck to be minimized, keep steady-state traffic within the real NVLink pairs (`GPU0 <-> GPU2`, `GPU1 <-> GPU3`) when multi-GPU work is involved, do not assume Ampere-only features, and do not lock in NCCL environment settings without measurement.

When performance-sensitive code is touched, prefer benchmark-backed or profiler-backed decisions over compatibility arguments. Nsight Systems should answer pipeline and overlap questions; Nsight Compute should answer hot-kernel questions. If a code change favors throughput at the cost of portability, generality, or API smoothness, that tradeoff is acceptable in this repository as long as it is deliberate and scoped to the hot path.
