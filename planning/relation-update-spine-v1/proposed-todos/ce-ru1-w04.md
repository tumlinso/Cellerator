# CE-RU1-W04: Execute N16 hybrid edge gradients in persistent edge order

Lane: `CE-RU1-L-WMMA`. Status on delivery: **planned**, not claimed or implemented.
Dependencies: `CE-RU1-W03`, `CE-RU1-N03`

## Goal and implementation direction

Launch repaired WMMA over prepared panels, extract valid score entries directly to authoritative physical edge slots, and execute sparse residual over disjoint slots. Reuse existing sparse half contraction arithmetic, not a second gradient semantic operation. Full sparse fallback computes the same rounded request. Score tile scratch and extraction are explicit stages; no claim of accumulator-residual fusion. Gradients remain f32 for update.

## Source and design inputs

Read `01_SCOPE_AND_DECISIONS.md`, `02_SEMANTIC_AND_NATIVE_CONTRACT.md`, `03_WMMA_AND_DECOMPOSITION.md`, `04_READINESS_AND_OWNERSHIP.md` and the relevant rows of `07_SOURCE_LEDGER.md` before editing. The comprehensive review is preserved in `basis/architecture-review.html`; user decisions recorded in `01` refine its implementation ordering. Current code is evidence, not a reason to preserve a worse abstraction. The declared paths are proposals where new and current owners where existing.

## Exclusive write scope

- `src/compute/architecture/providers/nvidia/sm70/edge_value_gradient/hybrid_gradient.cu`
- `include/Cellerator/compute/architecture/providers/nvidia/sm70/edge_value_gradient/hybrid_gradient.cuh`
- `tests/relation_update_spine_v1/hybrid_gradient_test.cu`

Read-only references may include the full affected implementation, tests and planning package. No writes to `.todo-orchestrator`, other planning packages, unrelated worktrees, `components/` or vendor/legacy trees are authorized by this task scope. Normal canonical task-evidence updates use the execution workflow only after authorization; never manually edit authority databases. A necessary additional source path requires a recorded scope handoff before editing.

## Acceptance conditions

1. Dense, sparse-only and mixed covers agree with independent quantized-input VJP; required dense and mixed fixtures actually launch WMMA.
2. Residual writes cannot overlap rectangular output ownership; every gradient slot is initialized each invocation.

## Required validation cases

- Forced sparse versus forced hybrid; exact cover partition; cancellation, nonfinite and empty cases.

Test targets and receipt contract are in `machine/acceptance_matrix.json` and `05_VALIDATION_AND_DEMO.md`. Code/test-authoring tasks must compile and test their available local contract. Their completion does not replace integrated sm70 gates owned by I05/I08. A device test that cannot execute is explicitly pending there, never a passed device claim. Every command receipt records exact HEAD, dirty state, command, exit, tool identity and tested numerical profile.

## Cold/hot path and resource rules

Cold preparation may allocate bounded scratch, maps, panel storage and events, charged to the preparation budget. Hot execution performs no topology discovery, global sorting, implicit canonicalization, event allocation or device-global fence. Buffers borrowed by asynchronous operations remain valid through completion; the returned read lease establishes a reader-completion edge before an in-place writer. Use pointer-plus-count structures and existing identity types; do not introduce a second training engine.

## Principal failure to avoid

Calling a sparse kernel while reporting WMMA is not a fallback success.

## Finish and handoff

Report changed paths, actual computation now reachable, tests run versus not run, and any remaining uncertainty. Preserve useful code in core ownership; no fake fallback or compatibility-only implementation may satisfy this task. Commit after the accepted task through the authorized execution workflow. Push and merge policy follows the later execution handoff, not this planning package. Do not activate follow-on work.
