---
slug: "split-workbench-runtime-backend"
status: "done"
execution: "closed"
owner: "codex"
created_at: "2026-04-18T13:13:00Z"
last_heartbeat_at: "2026-04-18T13:17:49Z"
last_reviewed_at: "2026-04-18T13:17:49Z"
stale_after_days: 14
objective: "Split src/workbench into adapter-owned surfaces plus a shared workbench/runtime backend without changing the current public API"
---

# Current Objective

## Summary
Split src/workbench into adapter-owned surfaces plus a shared workbench/runtime backend without changing the current public API.

## Quick Start
- Why this stream exists: the dataset workbench backend had drifted into a mixed UI-plus-runtime shape even though workbench is supposed to be an adapter-facing surface.
- In scope: move the shared backend implementation and private CUDA helper headers under src/workbench/runtime/, keep src/workbench/dataset_workbench.hh as the stable facade, and update build/docs to reflect the new ownership boundary.
- Out of scope / dependencies: no namespace rename, no public API redesign, and no deeper semantic rewrite of the runtime backend; a later pass can split runtime/pipeline.cu further by conversion, browse, and preprocess behavior.
- Required skills: todo-orchestrator.
- Required references: AGENTS.md, todos.md, todo-status.md, src/workbench/README.md, src/workbench/dataset_workbench.hh, src/workbench/dataset_workbench_main.cc, CMakeLists.txt, docs/architecture.qmd, docs/compute_and_models.qmd, docs/ingest_and_runtime.qmd.

## Planning Notes
- Keep the adapter-facing namespace and header stable first; move implementation ownership before attempting a wider API or behavior split.
- Preserve src/compute/preprocess as the reusable math/operator layer and treat workbench/runtime as orchestration over persisted dataset workflows.

## Assumptions
- The immediate goal is architectural separation, not a full behavior-level breakup of every runtime helper in one pass.
- Keeping cellerator::apps::workbench as the public namespace is the lowest-risk way to preserve ncurses and Python callers during the move.

## Suggested Skills
- `todo-orchestrator` - Track the adapter/runtime split as a resumable refactor with explicit verification notes.

## Useful Reference Files
- `src/workbench/README.md` - Workbench is the adapter-facing surface and now documents the runtime subdirectory split.
- `src/workbench/dataset_workbench.hh` - Stable public facade that remains the include point for UI and Python callers.
- `src/workbench/runtime/summary.cc` - Shared summary and metadata inspection backend moved out of the adapter root.
- `src/workbench/runtime/pipeline.cu` - Shared conversion, preprocess, finalize, and browse-cache backend moved out of the adapter root.
- `CMakeLists.txt` - Build graph now compiles the runtime backend from src/workbench/runtime/.

## Plan
- Move the non-UI summary/inspection implementation under src/workbench/runtime/.
- Move the dataset pipeline CUDA implementation and its private kernels under src/workbench/runtime/.
- Retarget cellerator_workbench at the new runtime paths and update architecture-facing docs.
- Rebuild the workbench library and focused runtime binaries to confirm the facade still links cleanly.

## Tasks
- [x] Move the summary backend into src/workbench/runtime.
- [x] Move the CUDA workflow backend and private kernels into src/workbench/runtime.
- [x] Update build and docs for the adapter/runtime split.
- [x] Rebuild and run focused workbench validation.

## Blockers
_None recorded yet._

## Progress Notes
- Moved the shared non-UI implementation from src/workbench/dataset_workbench.cc to src/workbench/runtime/summary.cc and the shared CUDA workflow implementation from src/workbench/dataset_workbench_cuda.cu to src/workbench/runtime/pipeline.cu.
- Moved the private workbench CUDA helper headers under src/workbench/runtime/kernels/ so the adapter root no longer owns those runtime-private files.
- Updated cellerator_workbench in CMakeLists.txt to compile the runtime backend from src/workbench/runtime/ and rewrote the workbench/architecture docs to describe the new adapter/runtime split.
- Built cellerator_workbench, datasetWorkbenchRuntimeTest, cellShardFirstFileFixtureTest, and celleratorWorkbench successfully; ran ./build/datasetWorkbenchRuntimeTest and ./build/cellShardFirstFileFixtureTest successfully.

## Next Actions
- Optional follow-up only: split src/workbench/runtime/pipeline.cu further by behavior once the adapter/runtime move has settled.

## Done Criteria
- The shared workbench backend no longer lives directly under src/workbench/ root implementation files.
- src/workbench/dataset_workbench.hh remains the stable public include and callers do not need namespace changes.
- The build graph compiles the runtime backend from src/workbench/runtime/.
- Focused workbench build and runtime validation pass after the move.
