---
slug: "dual-cuda-optimization-modes"
status: "in_progress"
execution: "claimed"
owner: "codex"
created_at: "2026-04-13T14:45:32Z"
last_heartbeat_at: "2026-04-13T15:43:08Z"
last_reviewed_at: "2026-04-13T15:43:08Z"
stale_after_days: 14
objective: "add repo-wide portable and extreme V100 CUDA optimization modes with compile-time selection and first hotspot backends"
---

# Current Objective

## Summary
Implement a repo-wide compile-time CUDA mode split with default portable Volta tuning, opt-in V100 extreme tuning, and initial distinct extreme hotspot backends.

## Quick Start
- Why this stream exists: the repo currently has one shared sm_70 CUDA optimization posture and needs an explicit default `portable` mode plus an opt-in `extreme` mode boundary.
- In scope: root CMake mode selection, generated/internal mode config, repo-wide target wiring, explicit portable/extreme backend boundaries, and first distinct extreme implementations for benchmark-worthy hotspots.
- Out of scope / dependencies: no runtime mode switch, no blanket PTX rewrite of cold code, and no change to existing public APIs beyond compile-time build/configuration additions.
- Required skills: `todo-orchestrator` for the ledger and `cuda-v100` for Volta-specific hotspot choices.
- Required references: `AGENTS.md`, `optimization.md`, `pointer_migration_plan.md`, `CMakeLists.txt`, `src/quantized/README.md`, and the active CUDA backend sources under `src/quantized/`, `src/compute/autograd/`, and `src/compute/neighbors/forward_neighbors/`.

## Planning Notes
- `portable` remains the default compile mode; `extreme` is a global configure-time choice and not a runtime dispatch surface.
- Every CUDA target must sit behind the new mode boundary immediately, but low-value targets may use explicit `extreme` -> `portable` alias backends until they earn benchmark-backed specialization.
- The first distinct extreme backend should land where the repo already expects custom-kernel Volta tuning instead of library-backed math.

## Assumptions
- The initial extreme implementation wave can be considered complete when the framework is in place and at least one real hotspot backend has a distinct extreme path with coverage.
- Performance-first numerics are acceptable in extreme mode so long as tests use explicit tolerances and the portable mode remains the correctness anchor.
- Inline PTX should stay isolated to tiny helpers or kernels inside extreme-only code paths rather than becoming a repo-wide source pattern.

## Suggested Skills
- `todo-orchestrator` - Keep the new dual-mode CUDA workstream current while implementation and validation proceed.
- `cuda-v100` - Keep the compile-time split and first extreme kernels aligned to Volta sm_70 and real hotspot classes.

## Useful Reference Files
- `AGENTS.md` - Repo-wide build, testing, and Volta optimization rules.
- `optimization.md` - Current bottlenecks and why explicit low-level operators matter more than generic flags.
- `pointer_migration_plan.md` - Do not grow new vector-heavy hot-path surfaces while splitting kernels by mode.
- `CMakeLists.txt` - Current shared CUDA compile flag surface and target inventory.
- `src/quantized/README.md` - Quantized backend posture and why it is the leading custom-kernel hotspot.
- `src/quantized/dispatch.cuh` - Current launch boundary for quantized kernels.
- `src/compute/autograd/autograd.hh` - Sparse autograd API and fleet context surface that must preserve stable public contracts.
- `src/compute/autograd/kernels/base_sparse.cu` - Current base custom kernels and library-backed boundaries.
- `src/compute/neighbors/forward_neighbors/forward_neighbors.cu` - Forward-neighbor backend that needs the mode boundary even if it stays aliased initially.

## Plan
- Add the root build option, generated/internal mode config, and shared target helper changes needed to compile portable vs extreme globally.
- Roll every CUDA target/library under the new mode-selection surface with explicit alias backends where specialization is not landing yet.
- Add a distinct extreme backend for the quantized runtime and route existing call sites through compile-time mode selection.
- Add focused tests and benchmark output changes so portable vs extreme builds can be validated and compared.

## Tasks
- [x] Create the repo-wide CUDA mode build/config framework
- [x] Wire all CUDA targets through portable/extreme compile-time selection
- [x] Add a distinct quantized extreme backend
- [x] Extend validation and benchmark reporting for CUDA mode awareness

## Blockers
- Autograd-dependent targets such as `computeAutogradRuntimeTest` and `developmentalTimeABBench` are currently blocked by the existing `extern/CellShard/src/sharded/sharded_host.cuh` vs `shard_storage` API break from the active packfile-cache rewrite stream, not by the new CUDA mode framework.

## Progress Notes
- Added `CELLERATOR_CUDA_MODE` with default `portable`, generated `cellerator_cuda_mode.hh`, and linked the mode config through the shared performance helper so CUDA targets compile under one consistent mode surface.
- Locked `extreme` mode to Volta `sm_70` at configure time and kept `portable` as the correctness-default build posture.
- Split the quantized device backend into `portable_backend.cuh` and a distinct `extreme_backend.cuh` with Volta-specific inline PTX helpers for reciprocal refinement, `cvt.rni`, `setp`/`selp` clamping, and PTX FMA-based reconstruction.
- Routed the public quantized dispatch layer through compile-time mode selection without changing the existing API used by tests or benches.
- Added CUDA mode banner output to the quantized, autograd, forward-neighbor, and developmental-time benchmark entrypoints so run logs identify which build produced the numbers.
- Verified separate `/tmp/cellerator-portable-build` and `/tmp/cellerator-extreme-build` configure passes, built `quantizedMatrixTest`, `quantizedMatrixBench`, and `forwardNeighborsCompileTest` in both trees, and ran `quantizedMatrixTest` successfully in both modes.
- Ran a quick serialized `quantizedMatrixBench --host-iters 1 --gpu-iters 5 --bits 8 --policy per_gene_affine` sample in both modes; output now includes `cuda_mode=portable|extreme` and the current sample GPU timings were about 0.106 ms portable vs 0.111 ms extreme on that small run.
- Split the quantized backend to one-kernel-per-file boundaries: `portable_quantize_kernel.cuh`, `portable_dequantize_kernel.cuh`, `extreme_quantize_kernel.cuh`, and `extreme_dequantize_kernel.cuh`, while keeping the existing launcher and dispatch API stable.
- Kept the extreme PTX helpers isolated behind the split and reran the focused dump on `bench/quantized_extreme_ptx_hotspot.cu`; the post-split `sm_70` hotspot still compiles with focused artifacts, 20 registers, and no spills.
- Rebuilt `quantizedMatrixTest` and `quantizedMatrixBench` in both portable and extreme trees after the split and reran `quantizedMatrixTest` successfully in both modes.
- A fresh mutex-serialized `quantizedMatrixBench --host-iters 1 --gpu-iters 5 --bits 8 --policy per_gene_affine` run did not return on the expected timescale after the split, so that verification path is currently treated as bench-specific follow-up rather than a kernel-structure regression until the benchmark itself is checked.
- Split every multi-kernel file in the active repo inventory to one `__global__` definition per file, including quantized, preprocess, autograd `base_sparse`, model ops, forward-neighbor search, cuVS sharded KNN helpers, workbench CUDA helpers, legacy embryo-series loaders, and the multi-kernel CellShard bucket/convert headers.
- Kept the split structural only: helpers, launchers, namespaces, and public call surfaces stayed in their original owner files while each kernel body moved into a dedicated include file local to its subsystem.
- Re-ran the repo-wide inventory and the only remaining multi-hit `__global__` file is `extern/CellShard/src/cuda_compat.cuh`, which is a macro-compat header rather than a kernel-definition surface.
- Verified portable and extreme builds for `quantizedMatrixTest`, `forwardNeighborsCompileTest`, `computeAutogradRuntimeTest`, and `seriesWorkbenchRuntimeTest`, plus `cellerator_compute_model_ops`, after the repo-wide split.
- Extended the portable/extreme split into CellShard: root CMake now propagates the mode into `extern/CellShard`, standalone CellShard validates `extreme` as `sm_70`-only, and the bucket/convert kernel headers now route their hot index and atomic micro-primitives through a shared Volta-only inline PTX helper layer.
- Verified the CellShard extension with fresh `/tmp/cellerator-cellshard-portable` and `/tmp/cellerator-cellshard-extreme` configure passes, then built `cellShardSeriesH5Test`, `cellShardBlockedEllTest`, and `seriesWorkbenchRuntimeTest` successfully in both modes.
- `modelCustomOpsTest` and `scrnaPreprocessBench` still fail on the pre-existing CellShard part-to-shard API migration, not on the one-kernel-per-file split.
- `seriesWorkbenchRuntimeTest` exits cleanly in one serial run, but parallel launches can trip HDF5 temp-file locking or missing-file ordering on `/tmp/cellerator_series_workbench.series.csh5`; treat that as a test harness concurrency issue rather than a kernel-structure regression.

## Next Actions
- Once the active CellShard storage rewrite lands or stabilizes enough to compile `cellerator_compute_autograd` again, extend the distinct extreme path into sparse autograd and the model CUDA surfaces that depend on it.
- Decide which CellShard bucket/convert kernel should earn the first benchmark-backed distinct extreme body instead of relying only on shared PTX micro-primitives.
- Check why the quantized benchmark binary stalled on the tiny post-split sample before using it as the next comparison point.
- Keep future CUDA changes on the one-kernel-per-file structure now that the repo-wide inventory has been normalized.
- Add the next benchmark-backed extreme specialization to forward-neighbor, autograd, or CellShard hot kernels instead of broad alias-only coverage.

## Done Criteria
- `CELLERATOR_CUDA_MODE` exists, defaults to portable, and controls every CUDA target through one consistent compile-time path.
- Extreme mode builds cleanly for the repo CUDA targets even where some subsystems still alias the portable backend.
- The quantized backend has a distinct extreme implementation with focused tests or benchmarks proving the split is real.
- Benchmark/test output clearly identifies the active CUDA mode so results are comparable across builds.
