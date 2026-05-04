# Active Objectives

## Summary
Use this file as the canonical index for substantial multi-step work.

## Shared Assumptions
- CellShardPreprocess owns the accelerated scRNA preprocessing backbone, native workbench implementation, and preprocessing benchmarks; Cellerator no longer keeps preprocessing compatibility wrappers.
- Native Cellerator GPU policy remains V100 `sm_70` and Blocked-ELL-first for Cellerator execution. CellShardPreprocess treats Blocked-ELL and Sliced-ELL as first-class preprocessing layouts, with compressed / CSR as fallback.

## Suggested Skills
- `todo-orchestrator`: maintain the resumable migration ledger while implementing the supplied plan.
- `bio-experiments`: preserve scRNA raw-count, QC, normalization, and double-processing semantics.
- `cuda`: keep the implementation aligned with native V100 sparse bio-data layout and hot-path constraints.

## Useful Reference Files
- `AGENTS.md`: repository structure, testing, Blocked-ELL, and Volta policy.
- `optimization.md`: current V100 bottleneck and sparse preprocessing guidance.
- `docs/pipeline/preprocess/README.md`: existing preprocessing documentation to update if behavior moves.
- `docs/pipeline/README.md`: pipeline-level documentation surface to keep in sync.

## Workstreams
- `cellerator-sparse-ml-layout`: in_progress / idle - Refactored Cellerator Core into contract-first matrix/runtime/quantized/interop layers, moved conversion and CUDA compute primitives under `src/compute`, updated CellShard shims, and kept `src/compute` as the sparse math/operator layer. Needs a decision on the standalone CellShard mask-groups exit-14 expectation.
- `cellshard-preprocess-gpu-biology-backbone`: done / closed - CellShardPreprocess owns native preprocessing APIs and benchmarks; Cellerator preprocessing API/implementation/benchmark targets have been removed.
- `cellshard-multi-assay-archive`: done / closed - Added the multi-assay archive foundation, pointer-first row-map validation, the biology semantics package-backed semantic checks, cshard POD descriptors, docs, and compile/runtime coverage while leaving CSPACK payloads single-assay.

## Global Blockers
_None recorded yet._

## Progress Notes
- Added the first CelleratorCore ownership slice: `Cellerator::core` now exposes
  sparse layout primitives/device views under `include/Cellerator/core`, and
  CellShard layout headers are compatibility shims over those types.
- Verified the CelleratorCore/CellShard wiring with CellShard package-consumer
  checks, Cellerator sparse/quantized tests, and a CellShardPreprocess build.
- Started `cellerator-sparse-ml-layout` from the supplied source layout plan. The intended first pass is behavior-preserving except for moving forward-neighbor policy/API ownership into the new the external neighbor-caller package.
- Checkpointed `cellerator-sparse-ml-layout`: moved sparse-operator/model-op code under `src/compute/ml`, moved shared host buffering to `src/compute/core`, moved cuVS/KNN scoring helpers under `src/compute/neighbors/scoring`, and removed Cellerator-owned forward-neighbor compatibility wrappers.
- Implemented `cellshard-multi-assay-archive`: CellShard now has measurement-agnostic assay descriptors and row-map helpers, Cellerator validates those semantics against the biology semantics package, and docs state that multiome execution uses coordinated single-assay CSPACK artifacts.
- Ran `todo-cleanup --partial` and cleared workstreams: dual-cuda-optimization-modes, cellshard-first-stable-release, cellshard-blocked-ell-ingest-runtime, cellshard-runtime-service-contract, quantized-blocked-ell-codecs, cellshard-user-metadata-annotations, gpu-prototype-ingest-blocked-ell, gpu-prototype-model-sparse-boundaries, gpu-prototype-neighbors-trajectory, blocked-ell-optimization-study, gpu-benchmark-sliced-preprocess-campaign, cellshard-hierarchy-reset, implement-derived-subset-and-reorder-materialization-for-cellshard-and-workbench.
- Started `cellshard-preprocess-gpu-biology-backbone` from the supplied implementation plan.
- Finished `cellshard-preprocess-gpu-biology-backbone`: moved Blocked-ELL/Sliced-ELL native preprocessing and CSR fallback ownership into CellShardPreprocess, moved preprocessing benchmarks there, and removed Cellerator preprocessing APIs and root targets.
- Checkpointed CelleratorCore layout migration: old `core/sparse` and `quantized` public paths were removed, Core owns matrix representation/runtime substrate/quantized packing under `include/Cellerator/core`, compute owns conversion and CUDA primitives, CellShard shims were updated, and focused Cellerator runtime checks passed. Standalone CellShard `cellShardMaskGroupsRuntimeTest` builds but exits 14 on its row-keep expectation.
- Checkpointed the matrix/compute boundary cleanup: `core/format` became `core/matrix`, conversion and bucket moved to `compute/matrix/convert`, `warp_reduce.cuh` moved under `src/compute/core/primitives`, and `cmake --build build -j 4` plus `./build/coreSparseLayoutRuntimeTest` passed.

## Next Actions
- Decide whether CellShard `cellShardMaskGroupsRuntimeTest` exit 14 is an existing CellShard behavior issue or should be fixed under `cellerator-sparse-ml-layout`.

## Done Criteria
- Every active workstream in `todos/` is reflected here with a current status.

## Historical Summary
- Recent completed work included Blocked-ELL persistence, real-data sparse replay benchmarking, quantize sparse operator kernels, workbench browse-cache updates, semantic the biology semantics package cleanup, and the initial pointer-first neighbor workspace refactor.
- Detailed historical workstream notes are preserved in git history rather than as active repo ledgers.
