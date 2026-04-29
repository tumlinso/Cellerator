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
- `cellerator-sparse-ml-layout`: in_progress / idle - Refactored Cellerator `src/compute` around sparse ML math contracts, kept compatibility wrappers, and moved forward-neighbor policy into CellShardNeighbors. Needs a final follow-up review/build after this checkpoint.
- `cellshard-preprocess-gpu-biology-backbone`: done / closed - CellShardPreprocess owns native preprocessing APIs and benchmarks; Cellerator preprocessing API/implementation/benchmark targets have been removed.
- `cellshard-multi-assay-archive`: done / closed - Added the multi-assay archive foundation, pointer-first row-map validation, cudaBioTypes-backed semantic checks, cshard POD descriptors, docs, and compile/runtime coverage while leaving CSPACK payloads single-assay.

## Global Blockers
_None recorded yet._

## Progress Notes
- Started `cellerator-sparse-ml-layout` from the supplied source layout plan. The intended first pass is behavior-preserving except for moving forward-neighbor policy/API ownership into the new `extern/CellShardNeighbors` submodule.
- Checkpointed `cellerator-sparse-ml-layout`: moved autograd/model-op code under `src/compute/ml`, moved shared host buffering to `src/compute/core`, moved cuVS/KNN scoring helpers under `src/compute/neighbors/scoring`, and moved forward-neighbor API/source into `extern/CellShardNeighbors` with Cellerator compatibility wrappers.
- Implemented `cellshard-multi-assay-archive`: CellShard now has measurement-agnostic assay descriptors and row-map helpers, Cellerator validates those semantics against cudaBioTypes, and docs state that multiome execution uses coordinated single-assay CSPACK artifacts.
- Ran `todo-cleanup --partial` and cleared workstreams: dual-cuda-optimization-modes, cellshard-first-stable-release, cellshard-blocked-ell-ingest-runtime, cellshard-runtime-service-contract, quantized-blocked-ell-codecs, cellshard-user-metadata-annotations, gpu-prototype-ingest-blocked-ell, gpu-prototype-model-sparse-boundaries, gpu-prototype-neighbors-trajectory, blocked-ell-optimization-study, gpu-benchmark-sliced-preprocess-campaign, cellshard-hierarchy-reset, implement-derived-subset-and-reorder-materialization-for-cellshard-and-workbench.
- Started `cellshard-preprocess-gpu-biology-backbone` from the supplied implementation plan.
- Finished `cellshard-preprocess-gpu-biology-backbone`: moved Blocked-ELL/Sliced-ELL native preprocessing and CSR fallback ownership into CellShardPreprocess, moved preprocessing benchmarks there, and removed Cellerator preprocessing APIs and root targets.

## Next Actions
- Resume by reviewing the CMake/include wiring after the source moves, then run a focused configure/build and the affected tests.

## Done Criteria
- Every active workstream in `todos/` is reflected here with a current status.

## Historical Summary
- Recent completed work included Blocked-ELL persistence, real-data sparse replay benchmarking, quantize autograd kernels, workbench browse-cache updates, semantic cudaBioTypes cleanup, and the initial pointer-first neighbor workspace refactor.
- Detailed historical workstream notes are preserved in git history rather than as active repo ledgers.
