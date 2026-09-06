# CE-RU1-I04: Complete and run the dual-origin regulatory learning demo

Lane: `CE-RU1-L-INTEGRATE`. Status on delivery: **planned**, not claimed or implemented.
Dependencies: `CE-RU1-I03`

## Goal and implementation direction

Bind the delivered prospective program to the integrated core API; spelling changes are allowed but preserve all assertions and document them. One 20-source/19-destination synthetic regulatory topology has a dense module, irregular residual, isolated source and empty row. Execute native/compiler forward and VJP equality, a delta update and gradient-step update, publication/read lease, stale-generation rejection and reuse. All new arithmetic stays in core, not demo-local CUDA kernels.

## Source and design inputs

Read `01_SCOPE_AND_DECISIONS.md`, `02_SEMANTIC_AND_NATIVE_CONTRACT.md`, `03_WMMA_AND_DECOMPOSITION.md`, `04_READINESS_AND_OWNERSHIP.md` and the relevant rows of `07_SOURCE_LEDGER.md` before editing. The comprehensive review is preserved in `basis/architecture-review.html`; user decisions recorded in `01` refine its implementation ordering. Current code is evidence, not a reason to preserve a worse abstraction. The declared paths are proposals where new and current owners where existing.

## Exclusive write scope

- `examples/relation_update_spine_v1`
- `docs/relation_update_spine_v1/demo.json`

Read-only references may include the full affected implementation, tests and planning package. No writes to `.todo-orchestrator`, other planning packages, unrelated worktrees, `components/` or vendor/legacy trees are authorized by this task scope. Normal canonical task-evidence updates use the execution workflow only after authorization; never manually edit authority databases. A necessary additional source path requires a recorded scope handoff before editing.

## Acceptance conditions

1. Normal demo links real Cellerator and sm70 implementations, not planning stubs or reference mode.
2. Exactly one topology preparation; both update styles and generations 1,2,3 demonstrated; real sparse/WMMA attribution printed.
3. The fixture is labelled synthetic and loss decrease is not a scientific learning-performance claim.

## Required validation cases

- Post-epic normal GPU demo, reference-only fixture run as separately labelled control.

Test targets and receipt contract are in `machine/acceptance_matrix.json` and `05_VALIDATION_AND_DEMO.md`. Code/test-authoring tasks must compile and test their available local contract. Their completion does not replace integrated sm70 gates owned by I05/I08. A device test that cannot execute is explicitly pending there, never a passed device claim. Every command receipt records exact HEAD, dirty state, command, exit, tool identity and tested numerical profile.

## Cold/hot path and resource rules

Cold preparation may allocate bounded scratch, maps, panel storage and events, charged to the preparation budget. Hot execution performs no topology discovery, global sorting, implicit canonicalization, event allocation or device-global fence. Buffers borrowed by asynchronous operations remain valid through completion; the returned read lease establishes a reader-completion edge before an in-place writer. Use pointer-plus-count structures and existing identity types; do not introduce a second training engine.

## Principal failure to avoid

The demo must not become a private implementation that lets unfinished core tasks pass.

## Finish and handoff

Report changed paths, actual computation now reachable, tests run versus not run, and any remaining uncertainty. Preserve useful code in core ownership; no fake fallback or compatibility-only implementation may satisfy this task. Commit after the accepted task through the authorized execution workflow. Push and merge policy follows the later execution handoff, not this planning package. Do not activate follow-on work.
