# Repository Guidelines

## Project Structure & Module Organization
`src/` holds active Cellerator code. Primary surfaces live under `src/compute/`, `src/ingest/`, `src/workbench/`, `src/models/`, `src/trajectory/`, `src/torch/`, and `src/quantized/`. `src/legacy/` is the quarantine area for old monolithic entrypoints or transitional code that has not been deleted yet. `tests/` contains compile and runtime checks, `bench/` contains performance benches, and `extern/CellShard/` is the storage submodule that Cellerator builds against.

`src/compute/` is the authoritative home for reusable low-level computation: preprocess operators, neighbor search, quantized helpers, Torch custom ops, and the sparse autograd building-block layer under `src/compute/autograd/`. That autograd surface is pointer-first and layout-explicit: `autograd.hh` defines raw-buffer contexts, scratch, cuSPARSE caches, and fleet helpers; `kernels/base_sparse.cu` holds the single-GPU CSR kernels and cuSPARSE-backed library paths; `kernels/dist_sparse.cu` holds the selected-slot distributed launch wrappers and explicit leader-merge reduction path for the real 4-GPU topology. `src/models/` is the header-first libtorch workflow surface. `src/models/developmental_time/` and `src/models/dense_reduce/` split each model into `*_dataloader.hh`, `*_model.hh`, `*_train.hh`, and `*_infer.hh` layers with umbrella headers (`developmentalTime.hh`, `denseReduction.hh`). Forward-neighbor retrieval indices live under `src/compute/neighbors/forward_neighbors/`, not under `src/models/`. `src/models/quantize/` learns per-gene quantization parameters and packs dense reconstructions into the Volta-oriented quantized backend under `src/quantized/`.

For substantial repo work, consult root `todos.md` first. Detailed workstream ledgers under `todos/` are optional and may not exist.

For custom gradients, the first intended handwritten gradient-calculator target is the quantized path. Treat that backend as the leading candidate for explicit backward logic before introducing broader custom-gradient machinery elsewhere.

## Build, Test, and Development Commands
Configure with `cmake -S . -B build` and build with `cmake --build build -j 4`. Useful targets include `scrnaPreprocessBench`, `cellShardSeriesH5Test`, `quantizedMatrixTest`, `seriesIngestCompileTest`, `seriesWorkbenchRuntimeTest`, `trajectoryCompileTest`, `trajectoryRuntimeTest`, `forwardNeighborsCompileTest`, `computeAutogradRuntimeTest`, `developmentalTimeCompileTest`, `denseReduceCompileTest`, `quantizeModelTest`, `torchBindingsCompileTest`, and `modelCustomOpsTest`. Run built tests directly because `ctest` is not configured, for example `./build/cellShardSeriesH5Test`, `./build/forwardNeighborsCompileTest`, `./build/computeAutogradRuntimeTest`, or `./build/quantizeModelTest`. Torch-enabled builds now prefer the source-built libtorch installed at `/usr/local` via `/usr/local/share/cmake/Torch`. If libtorch is unavailable, configure with `cmake -S . -B build -DCELLERATOR_ENABLE_TORCH_MODELS=OFF`; use `Torch_DIR` or `LIBTORCH_PATH` only when you intentionally need to override that default.

## Coding Style & Naming Conventions
Follow the existing C++17/CUDA17 style: 4-space indentation, opening braces on the same line, and standard-library names qualified with `std::` when they are used. Prefer `snake_case` for functions, variables, structs, and CLI flags; use short type aliases only when they improve readability. Match current file suffixes: `.cu` for CUDA translation units, `.cuh` for CUDA headers, `.cc`/`.hh` for C++ sources and headers. Keep storage/layout primitives in `extern/CellShard`; keep pipeline and model logic in Cellerator.

Bias new code toward HPC-first implementation choices. Performance on Volta `sm_70` takes priority over genericity, portability, or abstraction count in performance-sensitive code. Do not keep a standard-library abstraction in a hot path just because it is idiomatic C++ if a lower-level representation is measurably or obviously better for throughput, copy behavior, launch behavior, memory traffic, or layout control.

This repository is not trying to abstract low-level runtime building blocks away. Kernels, layouts, residency boundaries, scratch buffers, and launch structure are supposed to remain visible enough to optimize directly. If a “cleaner” abstraction hides the real cost model or blocks the best Volta path, it is the wrong fit for this codebase.

The repository-wide migration plan for removing vector-heavy hot-path code lives in `pointer_migration_plan.md` at the repo root. Read that file before changing any of the current heavy-violation subsystems or introducing a new GPU-facing buffer surface.

`std::vector` is not the default container for GPU-facing or copy-sensitive code, and it is not an acceptable steady-state container in hot paths. Prefer explicit contiguous buffers, pointer-plus-size interfaces, fixed-capacity or preallocated storage, trivially copyable structs, and layout choices that make host-device transfers and kernel access patterns obvious. Use raw pointers, `std::unique_ptr<T[]>`, aligned allocations, or library-owned buffers when they reduce hidden reallocations, iterator-heavy code, ownership ambiguity, or extra copies. Keep `std::vector`, `std::string`, `std::function`, stream-heavy I/O, and similar abstractions out of kernels, transfer boundaries, repeated batch assembly, merge scratch, and tight preprocessing loops unless they are clearly off the hot path or have been shown not to matter.

Do not introduce new `std::vector`-based public APIs for hot subsystems. When touching an existing vector-heavy hot path, bias the change toward the pointer-first end state described in `pointer_migration_plan.md` rather than preserving the current container surface.

Prefer structure-of-arrays over array-of-structures when access is columnar or warp-coalescing benefits. Preallocate aggressively, fuse passes when that removes HBM traffic, avoid allocator churn inside repeated workflows, and keep data resident on device whenever feasible. If a simpler but slower abstraction is retained, document the reason or the measurement that justified it.

For `src/models/`, preserve the current header-only libtorch pattern unless there is a clear build or compile-time reason to split it. New model workflows should follow the existing `dataloader` / `model` / `train` / `infer` breakdown and expose a single umbrella header per module.

## Testing Guidelines
Add tests under `tests/` beside the nearest feature area and name them after the unit or workflow being checked, for example `series_ingest_compile_test.cu`. Cover both compile-only integration points and small runtime checks when behavior can be exercised locally. For model work, prefer compile coverage for the umbrella header and a focused runtime test for loss, inference, or retrieval behavior when the path can run locally. For GPU-facing changes, build the affected target and run the corresponding binary; include the exact command in your PR notes.

Benchmark and profiler runs must be serialized across workers. Treat the repository-wide benchmark mutex as mandatory so concurrent agents do not skew GPU measurements. When adding or changing a benchmark binary under `bench/`, wire it through `bench/benchmark_mutex.hh` and do not bypass that lock in normal benchmark workflows.

Model-facing and compute-adjacent targets currently map to:
- `developmentalTimeCompileTest` for `src/models/developmental_time/`
- `denseReduceCompileTest` for `src/models/dense_reduce/`
- `forwardNeighborsCompileTest` for `src/compute/neighbors/forward_neighbors/`
- `quantizeModelTest` for `src/models/quantize/`
- `torchBindingsCompileTest` for `src/torch/` and CellShard-to-libtorch tensor export
- `computeAutogradRuntimeTest` for `src/compute/autograd/`
- `modelCustomOpsTest` for `src/compute/model_ops/`

## Model Design And Skills
When the task is deciding what model family, objective, latent structure, decoder, or loss should exist in `src/models/`, use `$v100-model-design` first. That includes new developmental-time models, latent reduction variants, multimodal or temporal extensions, and decisions about whether the work should stay in libtorch or begin in Python PyTorch with a later C++ path. Stay in that skill until the model choice, scaling posture, and any custom-op boundary are stable.

When the task moves from model choice to implementation constraints on the 4x V100 host, use `$cuda-v100` as the default path. That includes `sm_70` build assumptions, CUDA 12.x compatibility, memory fit, host-device staging, NCCL or DDP topology, quantized kernels, profiler-driven tuning, and any custom Torch CUDA extension boundary. Follow the skill's path-selection logic instead of treating CUDA work as generic GPU programming: choose the matching V100 path first, load only the relevant addendum or base reference, and optimize for the real bottleneck.

For repository-specific performance context, read `optimization.md` at the repo root before making or defending changes in hot paths. It documents the current subsystem-level bottlenecks, likely fixed-call overheads, V100-oriented optimization priorities, and the repo's bias toward explicit low-level building blocks over abstraction-heavy surfaces.

Sparse layout policy is now Blocked-ELL-first. Treat Blocked-ELL as the native sparse type for persisted execution, staging, and hot-path compute unless a subsystem is explicitly operating in a fallback compressed-only mode. Do not describe compressed / CSR as the default sparse representation in new code or docs unless that specific surface really is still fallback-only.

For migration work aimed at removing `std::vector` and abstraction from hot code, also read `pointer_migration_plan.md` at the repo root. It defines the subsystem ordering, target representations, and exit criteria for the pointer-first rewrite.

For manuscript planning, model framing, or conceptual discussion about regulatory dynamics, `docs/notes.txt` contains background notes from an unfinished separate project that may still be useful as reference context. Treat `docs/notes.txt` as optional background material rather than current manuscript source or repository ground truth unless the user explicitly asks to build from it.

If work around `docs/`, notes, or manuscript-like material turns into writing, figure generation, or citation support, use `$quarto-manuscript` rather than splitting those tasks across separate Quarto skills.

When code changes materially alter runtime, storage, ingest, execution-pack, or other pipeline behavior, update the corresponding documentation in `docs/` and any primary README surface that describes that behavior as part of the same change. Do not leave behavior documentation stale after the implementation lands.

If new model work in `src/models/` appears to need custom Torch ops, record the proposed op boundary in `custom_torch_ops.md` before implementing it and keep the op scope minimal. Prefer library-backed Torch, ATen, cuBLAS, cuSPARSE, or CUTLASS paths before adding handwritten CUDA.

For CUDA/C++ implementation work outside `src/models/`, default to `$cuda-v100` whenever kernel shape, memory fit, communication topology, HtoD staging, sparse layout, or profiler interpretation is material to the answer. Recommendations should explicitly say whether they are library-backed or custom-kernel, whether they assume `sm_70`, and what the dominant limiter is: PCIe, HBM traffic, occupancy, register pressure, launch overhead, or cross-GPU communication.

Do not preserve a higher-level abstraction purely for compatibility if it blocks the fastest reasonable Volta path. Prefer explicit data layouts, pointer-stable ownership, fused kernels, and pair-local communication patterns when they are the better performance choice.

## Commit & Pull Request Guidelines
Recent commits use short, lowercase summaries such as `move ingest to smellerator` and `reduce & time models`. Keep commit subjects brief, specific, and scoped to one change. PRs should explain the affected module, list build/test commands run, note any CUDA/HDF5/libtorch assumptions, and include benchmark deltas when touching kernels, ingest throughput, or preprocessing performance.

## Configuration Notes
CMake defaults to the local HPC SDK CUDA toolchain and `g++-12` host compiler when no override is supplied. Prefer explicit overrides with `CUDACXX` and `CUDAHOSTCXX` if you need a different toolchain.

For libtorch, prefer the source-built installation under `/usr/local` over a Python-packaged or PyTorch-bundled libtorch. That `/usr/local` build is the repository default because it is tuned for this host's V100/NVLink layout. Only point CMake at another libtorch with `Torch_DIR` or `LIBTORCH_PATH` when you are deliberately testing or debugging against a different build.

Model and quantization work should assume the repository is targeting Volta `sm_70` on Tesla V100 16 GB GPUs unless the task says otherwise. Treat PCIe as a bottleneck to be minimized, keep steady-state traffic within the real NVLink pairs (`GPU0 <-> GPU2`, `GPU1 <-> GPU3`) when multi-GPU work is involved, do not assume Ampere-only features, and do not lock in NCCL environment settings without measurement.

When performance-sensitive code is touched, prefer benchmark-backed or profiler-backed decisions over compatibility arguments. Nsight Systems should answer pipeline and overlap questions; Nsight Compute should answer hot-kernel questions. If a code change favors throughput at the cost of portability, generality, or API smoothness, that tradeoff is acceptable in this repository as long as it is deliberate and scoped to the hot path.
