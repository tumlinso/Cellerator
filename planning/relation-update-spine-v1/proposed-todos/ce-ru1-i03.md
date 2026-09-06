# CE-RU1-I03: Integrate runtime, providers, compiler and test targets

Lane: `CE-RU1-L-INTEGRATE`. Status on delivery: **planned**, not claimed or implemented.
Dependencies: `CE-RU1-N06`, `CE-RU1-R03`, `CE-RU1-W05`, `CE-RU1-F04`, `CE-RU1-V05`

## Goal and implementation direction

Merge the independent owned changes, wire actual core/provider/compiler symbols and all RU1 executable tests, and build with verified CUDA 12.x sm70. Use meaningful internal targets; do not stabilize the installed SDK. Resolve link and scope conflicts at the integration boundary, preserving every useful implementation. Confirm source-to-semantic and native requests reach identical math functions.

## Source and design inputs

Read `01_SCOPE_AND_DECISIONS.md`, `02_SEMANTIC_AND_NATIVE_CONTRACT.md`, `03_WMMA_AND_DECOMPOSITION.md`, `04_READINESS_AND_OWNERSHIP.md` and the relevant rows of `07_SOURCE_LEDGER.md` before editing. The comprehensive review is preserved in `basis/architecture-review.html`; user decisions recorded in `01` refine its implementation ordering. Current code is evidence, not a reason to preserve a worse abstraction. The declared paths are proposals where new and current owners where existing.

## Exclusive write scope

- `CMakeLists.txt`
- `src/compute/CMakeLists.txt`
- `src/runtime/CMakeLists.txt`
- `src/compiler/CMakeLists.txt`
- `tests/relation_update_spine_v1/CMakeLists.txt`
- `docs/relation_update_spine_v1/integration.json`

Read-only references may include the full affected implementation, tests and planning package. No writes to `.todo-orchestrator`, other planning packages, unrelated worktrees, `components/` or vendor/legacy trees are authorized by this task scope. Normal canonical task-evidence updates use the execution workflow only after authorization; never manually edit authority databases. A necessary additional source path requires a recorded scope handoff before editing.

## Acceptance conditions

1. Host and CUDA targets link without adding planning/contracts to production include paths.
2. Every advertised RU1 provider has an actual symbol and executable test; no uncompiled code-only completion.

## Required validation cases

- Host tests plus build of exact CTest inventory specified in the acceptance matrix.

Test targets and receipt contract are in `machine/acceptance_matrix.json` and `05_VALIDATION_AND_DEMO.md`. Code/test-authoring tasks must compile and test their available local contract. Their completion does not replace integrated sm70 gates owned by I05/I08. A device test that cannot execute is explicitly pending there, never a passed device claim. Every command receipt records exact HEAD, dirty state, command, exit, tool identity and tested numerical profile.

## Cold/hot path and resource rules

Cold preparation may allocate bounded scratch, maps, panel storage and events, charged to the preparation budget. Hot execution performs no topology discovery, global sorting, implicit canonicalization, event allocation or device-global fence. Buffers borrowed by asynchronous operations remain valid through completion; the returned read lease establishes a reader-completion edge before an in-place writer. Use pointer-plus-count structures and existing identity types; do not introduce a second training engine.

## Principal failure to avoid

An INTERFACE target or installed header is not an implemented computational capability.

## Finish and handoff

Report changed paths, actual computation now reachable, tests run versus not run, and any remaining uncertainty. Preserve useful code in core ownership; no fake fallback or compatibility-only implementation may satisfy this task. Commit after the accepted task through the authorized execution workflow. Push and merge policy follows the later execution handoff, not this planning package. Do not activate follow-on work.
