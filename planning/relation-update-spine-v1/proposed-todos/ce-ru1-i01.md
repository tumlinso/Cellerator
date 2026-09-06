# CE-RU1-I01: Revalidate the completed spine and execution baseline

Lane: `CE-RU1-L-INTEGRATE`. Status on delivery: **planned**, not claimed or implemented.
Dependencies: No implementation task prerequisite; separate execution authorization remains required.

## Goal and implementation direction

Reobserve Cellerator and Todo independently; preserve every unrelated branch and pending patch. Verify the actual N1 spine test targets, one-owner-stream contract, FMP1 and CTP1 bindings, CUDA 12.x toolchain and sm70 device through source and an execution baseline. Inventory callers of old native training and gradient-publication IR before assigning disposition. Record current failures rather than silently repairing unrelated projects.

## Source and design inputs

Read `01_SCOPE_AND_DECISIONS.md`, `02_SEMANTIC_AND_NATIVE_CONTRACT.md`, `03_WMMA_AND_DECOMPOSITION.md`, `04_READINESS_AND_OWNERSHIP.md` and the relevant rows of `07_SOURCE_LEDGER.md` before editing. The comprehensive review is preserved in `basis/architecture-review.html`; user decisions recorded in `01` refine its implementation ordering. Current code is evidence, not a reason to preserve a worse abstraction. The declared paths are proposals where new and current owners where existing.

## Exclusive write scope

- `docs/relation_update_spine_v1/baseline.json`
- `docs/relation_update_spine_v1/source_disposition.json`

Read-only references may include the full affected implementation, tests and planning package. No writes to `.todo-orchestrator`, other planning packages, unrelated worktrees, `components/` or vendor/legacy trees are authorized by this task scope. Normal canonical task-evidence updates use the execution workflow only after authorization; never manually edit authority databases. A necessary additional source path requires a recorded scope handoff before editing.

## Acceptance conditions

1. Record exact source hashes, worktree and authority cursor, CUDA/compiler/device identity, commands, exits and pre-existing failures.
2. No historical campaign is reopened; useful kernels and all on-path consumers have a named preserve/move/replace/retire decision.

## Required validation cases

- Existing Semantic Spine N1 acceptance and graph test discovery; baseline on the required V100 when hardware lease is available.

Test targets and receipt contract are in `machine/acceptance_matrix.json` and `05_VALIDATION_AND_DEMO.md`. Code/test-authoring tasks must compile and test their available local contract. Their completion does not replace integrated sm70 gates owned by I05/I08. A device test that cannot execute is explicitly pending there, never a passed device claim. Every command receipt records exact HEAD, dirty state, command, exit, tool identity and tested numerical profile.

## Cold/hot path and resource rules

Cold preparation may allocate bounded scratch, maps, panel storage and events, charged to the preparation budget. Hot execution performs no topology discovery, global sorting, implicit canonicalization, event allocation or device-global fence. Buffers borrowed by asynchronous operations remain valid through completion; the returned read lease establishes a reader-completion edge before an in-place writer. Use pointer-plus-count structures and existing identity types; do not introduce a second training engine.

## Principal failure to avoid

Obsolete run pointers and historical artifacts are not empty authority and not permission to clean it.

## Finish and handoff

Report changed paths, actual computation now reachable, tests run versus not run, and any remaining uncertainty. Preserve useful code in core ownership; no fake fallback or compatibility-only implementation may satisfy this task. Commit after the accepted task through the authorized execution workflow. Push and merge policy follows the later execution handoff, not this planning package. Do not activate follow-on work.
