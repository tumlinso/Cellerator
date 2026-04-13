# Active Objectives

## Summary
Use this file as the canonical index for substantial multi-step work.

## Shared Assumptions
_None recorded yet._

## Suggested Skills
_None recorded yet._

## Useful Reference Files
_None recorded yet._

## Workstreams
- `cellshard-core-partition-rename` | status: done | owner: codex | file: `todos/cellshard-core-partition-rename.md` | objective: rename CellShard core storage runtime and schema terminology from part to partition without compatibility shims
- `dual-cuda-optimization-modes` | status: in_progress | owner: codex | file: `todos/dual-cuda-optimization-modes.md` | objective: add repo-wide portable and extreme V100 CUDA optimization modes with compile-time selection and first hotspot backends

## Global Blockers
_None recorded yet._

## Progress Notes
- Ran `todo-cleanup --partial` and cleared workstreams: cellshard-first-class-build-export-python, public-omics-shortlist-manuscript-benchmark-seed, cellshard-debug-thread, cellshard-packfile-cache-rewrite.
- Finished `cellshard-core-partition-rename`, including the remaining CellShard sharded/HDF5 rename sweep plus focused build and runtime verification.

## Next Actions
- Continue `dual-cuda-optimization-modes`, or run `todo-cleanup --partial` to remove the completed partition-rename ledger.

## Done Criteria
- Every active workstream in `todos/` is reflected here with a current status.

## Historical Summary
- Recent completed work included Blocked-ELL persistence, real-data sparse replay benchmarking, quantize autograd kernels, workbench browse-cache updates, semantic cudaBioTypes cleanup, and the initial pointer-first neighbor workspace refactor.
- Detailed historical workstream notes are preserved in git history rather than as active repo ledgers.
