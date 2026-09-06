# CE-RU1-V02: Author GPU numerical and exact-cover acceptance

Lane: `CE-RU1-L-VERIFY`. Status on delivery: **planned**, not claimed or implemented.
Dependencies: `CE-RU1-V01`, `CE-RU1-N03`, `CE-RU1-W04`

## Goal and implementation direction

Compare actual GPU sparse, forced hybrid and mixed covers with independent oracles on non-square supports, shuffled physical mappings, empty/isolated rows and high-degree stress. Separate full-f32 accumulation tolerance from quantized operand error. Check every logical gradient, not aggregate loss alone. Register executable assertions for WMMA launch attribution and residual coverage.

## Source and design inputs

Read `01_SCOPE_AND_DECISIONS.md`, `02_SEMANTIC_AND_NATIVE_CONTRACT.md`, `03_WMMA_AND_DECOMPOSITION.md`, `04_READINESS_AND_OWNERSHIP.md` and the relevant rows of `07_SOURCE_LEDGER.md` before editing. The comprehensive review is preserved in `basis/architecture-review.html`; user decisions recorded in `01` refine its implementation ordering. Current code is evidence, not a reason to preserve a worse abstraction. The declared paths are proposals where new and current owners where existing.

## Exclusive write scope

- `tests/relation_update_spine_v1/gpu_numerics_test.cu`

Read-only references may include the full affected implementation, tests and planning package. No writes to `.todo-orchestrator`, other planning packages, unrelated worktrees, `components/` or vendor/legacy trees are authorized by this task scope. Normal canonical task-evidence updates use the execution workflow only after authorization; never manually edit authority databases. A necessary additional source path requires a recorded scope handoff before editing.

## Acceptance conditions

1. All per-edge and dense results meet declared policy-specific tolerances.
2. Forced hybrid dense/mixed fixtures record actual WMMA execution; fallback fixtures record actual sparse execution.

## Required validation cases

- N16 arbitrary input rebinding; residue mask/padding sentinels; instruction legality adversarial fixture.

Test targets and receipt contract are in `machine/acceptance_matrix.json` and `05_VALIDATION_AND_DEMO.md`. Code/test-authoring tasks must compile and test their available local contract. Their completion does not replace integrated sm70 gates owned by I05/I08. A device test that cannot execute is explicitly pending there, never a passed device claim. Every command receipt records exact HEAD, dirty state, command, exit, tool identity and tested numerical profile.

## Cold/hot path and resource rules

Cold preparation may allocate bounded scratch, maps, panel storage and events, charged to the preparation budget. Hot execution performs no topology discovery, global sorting, implicit canonicalization, event allocation or device-global fence. Buffers borrowed by asynchronous operations remain valid through completion; the returned read lease establishes a reader-completion edge before an in-place writer. Use pointer-plus-count structures and existing identity types; do not introduce a second training engine.

## Principal failure to avoid

Loss reduction alone cannot detect gradients assigned to the wrong biological edges.

## Finish and handoff

Report changed paths, actual computation now reachable, tests run versus not run, and any remaining uncertainty. Preserve useful code in core ownership; no fake fallback or compatibility-only implementation may satisfy this task. Commit after the accepted task through the authorized execution workflow. Push and merge policy follows the later execution handoff, not this planning package. Do not activate follow-on work.
