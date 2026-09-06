# CE-RU1-N03: Prepare physical edge gradients and full-f32 sparse VJP

Lane: `CE-RU1-L-NATIVE`. Status on delivery: **planned**, not claimed or implemented.
Dependencies: `CE-RU1-N02`

## Goal and implementation direction

Prepare logical-to-physical and inverse edge maps once, using FMP1 positions as the default authoritative order. Implement or adapt a bounded N16 f32-operand sparse dot path with f32 accumulation. Keep the existing half-operand sparse contraction for the mixed profile. Bind input/cotangent versions and output plane identity, check overlap/capacity, and stamp gradient production only after accepted enqueue. Reuse maps for observation without applying them every hot call.

## Source and design inputs

Read `01_SCOPE_AND_DECISIONS.md`, `02_SEMANTIC_AND_NATIVE_CONTRACT.md`, `03_WMMA_AND_DECOMPOSITION.md`, `04_READINESS_AND_OWNERSHIP.md` and the relevant rows of `07_SOURCE_LEDGER.md` before editing. The comprehensive review is preserved in `basis/architecture-review.html`; user decisions recorded in `01` refine its implementation ordering. Current code is evidence, not a reason to preserve a worse abstraction. The declared paths are proposals where new and current owners where existing.

## Exclusive write scope

- `src/compute/operation/prepared_relation.cu`
- `src/compute/architecture/providers/nvidia/sm70/edge_value_gradient/relation_gradient.cu`
- `include/Cellerator/compute/architecture/providers/nvidia/sm70/edge_value_gradient/relation_gradient.cuh`
- `tests/relation_update_spine_v1/sparse_gradient_test.cu`

Read-only references may include the full affected implementation, tests and planning package. No writes to `.todo-orchestrator`, other planning packages, unrelated worktrees, `components/` or vendor/legacy trees are authorized by this task scope. Normal canonical task-evidence updates use the execution workflow only after authorization; never manually edit authority databases. A necessary additional source path requires a recorded scope handoff before editing.

## Acceptance conditions

1. Every logical edge has exactly one physical gradient slot and inverse mapping; missing/duplicate endpoints remain the established explicit rejection.
2. Full-f32 profile executes without half conversion; mixed-profile sparse result uses exactly the same operand quantization as WMMA.

## Required validation cases

- Distinct edge sentinels, randomized logical input ordering, independent VJP, gradient input-version mismatch.

Test targets and receipt contract are in `machine/acceptance_matrix.json` and `05_VALIDATION_AND_DEMO.md`. Code/test-authoring tasks must compile and test their available local contract. Their completion does not replace integrated sm70 gates owned by I05/I08. A device test that cannot execute is explicitly pending there, never a passed device claim. Every command receipt records exact HEAD, dirty state, command, exit, tool identity and tested numerical profile.

## Cold/hot path and resource rules

Cold preparation may allocate bounded scratch, maps, panel storage and events, charged to the preparation budget. Hot execution performs no topology discovery, global sorting, implicit canonicalization, event allocation or device-global fence. Buffers borrowed by asynchronous operations remain valid through completion; the returned read lease establishes a reader-completion edge before an in-place writer. Use pointer-plus-count structures and existing identity types; do not introduce a second training engine.

## Principal failure to avoid

Gradient arithmetic does not depend on w, but its relationship to a forward snapshot still must not be silently mixed.

## Finish and handoff

Report changed paths, actual computation now reachable, tests run versus not run, and any remaining uncertainty. Preserve useful code in core ownership; no fake fallback or compatibility-only implementation may satisfy this task. Commit after the accepted task through the authorized execution workflow. Push and merge policy follows the later execution handoff, not this planning package. Do not activate follow-on work.
