# CE-RU1-C02: Define prepared value, gradient and reader lifetime contracts

Lane: `CE-RU1-L-CORE`. Status on delivery: **planned**, not claimed or implemented.
Dependencies: `CE-RU1-C01`

## Goal and implementation direction

Evolve prepared_relation_pair. Define caller-owned gradient/delta buffers tagged by topology, epoch, physical order and count; separate input identity/version from raw pointer rebinding. Add bounded gradient preparation/forcing options and reusable scratch budgets. Define one external read lease at a time for v1, usable sequentially on any same-device consumer stream, backed by precreated ready/done events. No shared host-thread calls or other-stream relation execution. Publication is observable readiness, not historical value storage.

## Source and design inputs

Read `01_SCOPE_AND_DECISIONS.md`, `02_SEMANTIC_AND_NATIVE_CONTRACT.md`, `03_WMMA_AND_DECOMPOSITION.md`, `04_READINESS_AND_OWNERSHIP.md` and the relevant rows of `07_SOURCE_LEDGER.md` before editing. The comprehensive review is preserved in `basis/architecture-review.html`; user decisions recorded in `01` refine its implementation ordering. Current code is evidence, not a reason to preserve a worse abstraction. The declared paths are proposals where new and current owners where existing.

## Exclusive write scope

- `include/Cellerator/compute/operation/relation_update.hh`
- `include/Cellerator/compute/operation/prepared_relation.hh`
- `docs/relation_update_spine_v1/lifetime.md`

Read-only references may include the full affected implementation, tests and planning package. No writes to `.todo-orchestrator`, other planning packages, unrelated worktrees, `components/` or vendor/legacy trees are authorized by this task scope. Normal canonical task-evidence updates use the execution workflow only after authorization; never manually edit authority databases. A necessary additional source path requires a recorded scope handoff before editing.

## Acceptance conditions

1. State transitions, invalidation, generation overflow, lease replay, reader-return and teardown behavior are complete.
2. Update and publication share one mechanism; delta add and gradient-step convenience do not create separate storage owners.
3. Public declaration sketch and demo agree; no declaration is installed without real implementation.

## Required validation cases

- State-machine table covers uninitialized, published, borrowed, returned, updating and poisoned states.

Test targets and receipt contract are in `machine/acceptance_matrix.json` and `05_VALIDATION_AND_DEMO.md`. Code/test-authoring tasks must compile and test their available local contract. Their completion does not replace integrated sm70 gates owned by I05/I08. A device test that cannot execute is explicitly pending there, never a passed device claim. Every command receipt records exact HEAD, dirty state, command, exit, tool identity and tested numerical profile.

## Cold/hot path and resource rules

Cold preparation may allocate bounded scratch, maps, panel storage and events, charged to the preparation budget. Hot execution performs no topology discovery, global sorting, implicit canonicalization, event allocation or device-global fence. Buffers borrowed by asynchronous operations remain valid through completion; the returned read lease establishes a reader-completion edge before an in-place writer. Use pointer-plus-count structures and existing identity types; do not introduce a second training engine.

## Principal failure to avoid

Waiting for producer readiness does not protect a reader from the next in-place write.

## Finish and handoff

Report changed paths, actual computation now reachable, tests run versus not run, and any remaining uncertainty. Preserve useful code in core ownership; no fake fallback or compatibility-only implementation may satisfy this task. Commit after the accepted task through the authorized execution workflow. Push and merge policy follows the later execution handoff, not this planning package. Do not activate follow-on work.
