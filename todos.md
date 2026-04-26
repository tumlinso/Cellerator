# Active Objectives

## Summary
Use this file as the canonical index for substantial multi-step work.

## Shared Assumptions
- MosaiCell owns the accelerated scRNA preprocessing backbone and workbench implementation; Cellerator keeps compatibility wrappers.
- Native GPU policy remains V100 `sm_70`, Blocked-ELL-first persisted execution, fp16 steady-state matrix values, and fp32 accumulators for QC and summary statistics.

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
- `mosaicell-gpu-biology-backbone`: partial / active - MosaiCell external target, smaller native preprocessing/runtime targets, validation delegation, docs, and build/test coverage are in place; remaining Cellerator metadata/finalize/browse-cache compatibility code still needs extraction behind MosaiCell-owned pointer-first APIs.

## Global Blockers
_None recorded yet._

## Progress Notes
- Ran `todo-cleanup --partial` and cleared workstreams: dual-cuda-optimization-modes, cellshard-first-stable-release, cellshard-blocked-ell-ingest-runtime, cellshard-runtime-service-contract, quantized-blocked-ell-codecs, cellshard-user-metadata-annotations, gpu-prototype-ingest-blocked-ell, gpu-prototype-model-sparse-boundaries, gpu-prototype-neighbors-trajectory, blocked-ell-optimization-study, gpu-benchmark-sliced-preprocess-campaign, cellshard-hierarchy-reset, implement-derived-subset-and-reorder-materialization-for-cellshard-and-workbench.
- Started `mosaicell-gpu-biology-backbone` from the supplied implementation plan.
- Reopened `mosaicell-gpu-biology-backbone` after clarifying that the smaller MosaiCell target builds, but not all preprocessing orchestration source has moved out of Cellerator yet; GPU runtime tests build but could not run on this no-device environment.

## Next Actions
- Continue extracting remaining preprocessing orchestration into MosaiCell-owned targets before thinning the Cellerator compatibility layer further.

## Done Criteria
- Every active workstream in `todos/` is reflected here with a current status.

## Historical Summary
- Recent completed work included Blocked-ELL persistence, real-data sparse replay benchmarking, quantize autograd kernels, workbench browse-cache updates, semantic cudaBioTypes cleanup, and the initial pointer-first neighbor workspace refactor.
- Detailed historical workstream notes are preserved in git history rather than as active repo ledgers.
