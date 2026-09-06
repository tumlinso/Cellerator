# CE-RU1-I05: Run required V100 correctness and sanitizer acceptance

Lane: `CE-RU1-L-INTEGRATE`. Status on delivery: **planned**, not claimed or implemented.
Dependencies: `CE-RU1-I04`

## Goal and implementation direction

Acquire the existing project GPU lease, select an actual sm70 device and run all host/device suites, demo and relevant N1 regressions. Run Compute Sanitizer memcheck on all device binaries and racecheck/synccheck on packed WMMA and lifecycle tests as supported. Capture commands, exits, source hashes, GPU UUID and tool versions. Missing tooling/hardware is a blocked acceptance, not a pass. Distinguish sanitizer coverage from synchronization proof.

## Source and design inputs

Read `01_SCOPE_AND_DECISIONS.md`, `02_SEMANTIC_AND_NATIVE_CONTRACT.md`, `03_WMMA_AND_DECOMPOSITION.md`, `04_READINESS_AND_OWNERSHIP.md` and the relevant rows of `07_SOURCE_LEDGER.md` before editing. The comprehensive review is preserved in `basis/architecture-review.html`; user decisions recorded in `01` refine its implementation ordering. Current code is evidence, not a reason to preserve a worse abstraction. The declared paths are proposals where new and current owners where existing.

## Exclusive write scope

- `docs/relation_update_spine_v1/evidence/gpu_acceptance.json`
- `docs/relation_update_spine_v1/evidence/sanitizer.json`
- `docs/relation_update_spine_v1/evidence/commands.json`

Read-only references may include the full affected implementation, tests and planning package. No writes to `.todo-orchestrator`, other planning packages, unrelated worktrees, `components/` or vendor/legacy trees are authorized by this task scope. Normal canonical task-evidence updates use the execution workflow only after authorization; never manually edit authority databases. A necessary additional source path requires a recorded scope handoff before editing.

## Acceptance conditions

1. Every required test executed with zero failures and no skipped device substitution.
2. Required memcheck has zero reported errors; racecheck/synccheck findings are resolved or an explicit bounded tool limitation blocks relevant completion.
3. No root closure on build-only evidence.

## Required validation cases

- Host contract/bridge; GPU numerical/binding/lifecycle; normal demo; N1 graph regressions.

Test targets and receipt contract are in `machine/acceptance_matrix.json` and `05_VALIDATION_AND_DEMO.md`. Code/test-authoring tasks must compile and test their available local contract. Their completion does not replace integrated sm70 gates owned by I05/I08. A device test that cannot execute is explicitly pending there, never a passed device claim. Every command receipt records exact HEAD, dirty state, command, exit, tool identity and tested numerical profile.

## Cold/hot path and resource rules

Cold preparation may allocate bounded scratch, maps, panel storage and events, charged to the preparation budget. Hot execution performs no topology discovery, global sorting, implicit canonicalization, event allocation or device-global fence. Buffers borrowed by asynchronous operations remain valid through completion; the returned read lease establishes a reader-completion edge before an in-place writer. Use pointer-plus-count structures and existing identity types; do not introduce a second training engine.

## Principal failure to avoid

Only final integrated hardware evidence establishes the new executable closure.

## Finish and handoff

Report changed paths, actual computation now reachable, tests run versus not run, and any remaining uncertainty. Preserve useful code in core ownership; no fake fallback or compatibility-only implementation may satisfy this task. Commit after the accepted task through the authorized execution workflow. Push and merge policy follows the later execution handoff, not this planning package. Do not activate follow-on work.
