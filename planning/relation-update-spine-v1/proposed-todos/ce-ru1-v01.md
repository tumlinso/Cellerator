# CE-RU1-V01: Author independent VJP and storage-rounding oracles

Lane: `CE-RU1-L-VERIFY`. Status on delivery: **planned**, not claimed or implemented.
Dependencies: `CE-RU1-I02`

## Goal and implementation direction

Create independent logical-edge CPU loops, not a call back into production maps. Check forward, transpose, scalar edge VJP and f16 update rounding. Use finite differences on the continuous real-valued extension of stored weights with quantization disabled during perturbation, then test quantized update separately. Check mixed gradient against exactly rounded operands and report discrepancy from full-f32 VJP separately. Reuse the demo fixture data but not its implementation as the sole oracle.

## Source and design inputs

Read `01_SCOPE_AND_DECISIONS.md`, `02_SEMANTIC_AND_NATIVE_CONTRACT.md`, `03_WMMA_AND_DECOMPOSITION.md`, `04_READINESS_AND_OWNERSHIP.md` and the relevant rows of `07_SOURCE_LEDGER.md` before editing. The comprehensive review is preserved in `basis/architecture-review.html`; user decisions recorded in `01` refine its implementation ordering. Current code is evidence, not a reason to preserve a worse abstraction. The declared paths are proposals where new and current owners where existing.

## Exclusive write scope

- `tests/relation_update_spine_v1/reference_math.hh`
- `tests/relation_update_spine_v1/reference_test.cc`

Read-only references may include the full affected implementation, tests and planning package. No writes to `.todo-orchestrator`, other planning packages, unrelated worktrees, `components/` or vendor/legacy trees are authorized by this task scope. Normal canonical task-evidence updates use the execution workflow only after authorization; never manually edit authority databases. A necessary additional source path requires a recorded scope handoff before editing.

## Acceptance conditions

1. Adjoint and finite-difference checks catch transposed axes, missing K sums and wrong edge mapping.
2. Zero/isolated support and deterministic f16 rounding cases are independent of GPU implementation.

## Required validation cases

- N1 and N16; high-bit identity differences; stable fixed seeds; signed weights and cancellation.

Test targets and receipt contract are in `machine/acceptance_matrix.json` and `05_VALIDATION_AND_DEMO.md`. Code/test-authoring tasks must compile and test their available local contract. Their completion does not replace integrated sm70 gates owned by I05/I08. A device test that cannot execute is explicitly pending there, never a passed device claim. Every command receipt records exact HEAD, dirty state, command, exit, tool identity and tested numerical profile.

## Cold/hot path and resource rules

Cold preparation may allocate bounded scratch, maps, panel storage and events, charged to the preparation budget. Hot execution performs no topology discovery, global sorting, implicit canonicalization, event allocation or device-global fence. Buffers borrowed by asynchronous operations remain valid through completion; the returned read lease establishes a reader-completion edge before an in-place writer. Use pointer-plus-count structures and existing identity types; do not introduce a second training engine.

## Principal failure to avoid

Finite differences through a half-rounded parameter store produce misleading zero derivatives.

## Finish and handoff

Report changed paths, actual computation now reachable, tests run versus not run, and any remaining uncertainty. Preserve useful code in core ownership; no fake fallback or compatibility-only implementation may satisfy this task. Commit after the accepted task through the authorized execution workflow. Push and merge policy follows the later execution handoff, not this planning package. Do not activate follow-on work.
