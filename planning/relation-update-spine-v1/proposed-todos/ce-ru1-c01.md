# CE-RU1-C01: Specify relation calculus and explicit arithmetic profiles

Lane: `CE-RU1-L-CORE`. Status on delivery: **planned**, not claimed or implemented.
Dependencies: `CE-RU1-I01`

## Goal and implementation direction

Extend the canonical relation semantics with a small calculus descriptor, not a generic graph. Specify scalar support VJP, orientation and cotangent axes, overwrite gradient output, f32 accumulation, and two explicit contraction profiles: f32 operands, or f32 operands rounded once to binary16 nearest-even before either sparse or WMMA multiplication. Retain N1 public semantics. An update is a separately ordered effect with old/new generation bindings; a compiler or candidate identity is not mathematical equivalence.

## Source and design inputs

Read `01_SCOPE_AND_DECISIONS.md`, `02_SEMANTIC_AND_NATIVE_CONTRACT.md`, `03_WMMA_AND_DECOMPOSITION.md`, `04_READINESS_AND_OWNERSHIP.md` and the relevant rows of `07_SOURCE_LEDGER.md` before editing. The comprehensive review is preserved in `basis/architecture-review.html`; user decisions recorded in `01` refine its implementation ordering. Current code is evidence, not a reason to preserve a worse abstraction. The declared paths are proposals where new and current owners where existing.

## Exclusive write scope

- `include/Cellerator/compute/operation/relation_calculus.hh`
- `docs/relation_update_spine_v1/calculus.md`

Read-only references may include the full affected implementation, tests and planning package. No writes to `.todo-orchestrator`, other planning packages, unrelated worktrees, `components/` or vendor/legacy trees are authorized by this task scope. Normal canonical task-evidence updates use the execution workflow only after authorization; never manually edit authority databases. A necessary additional source path requires a recorded scope handoff before editing.

## Acceptance conditions

1. Native and compiler-origin descriptors compare every mathematical field while excluding runtime bindings/provenance.
2. Scalar contraction split-K sums partials; no edge-channel concatenation or implicit normalization is introduced.
3. Mixed-profile semantics and straight-line real-valued VJP versus storage rounding are explicitly distinguished.

## Required validation cases

- Same extents/different biological axes; orientation reversal; profile mismatch; unknown effect kind; edge-channel rejection.

Test targets and receipt contract are in `machine/acceptance_matrix.json` and `05_VALIDATION_AND_DEMO.md`. Code/test-authoring tasks must compile and test their available local contract. Their completion does not replace integrated sm70 gates owned by I05/I08. A device test that cannot execute is explicitly pending there, never a passed device claim. Every command receipt records exact HEAD, dirty state, command, exit, tool identity and tested numerical profile.

## Cold/hot path and resource rules

Cold preparation may allocate bounded scratch, maps, panel storage and events, charged to the preparation budget. Hot execution performs no topology discovery, global sorting, implicit canonicalization, event allocation or device-global fence. Buffers borrowed by asynchronous operations remain valid through completion; the returned read lease establishes a reader-completion edge before an in-place writer. Use pointer-plus-count structures and existing identity types; do not introduce a second training engine.

## Principal failure to avoid

An f16 provider must never silently round operands requested as full f32.

## Finish and handoff

Report changed paths, actual computation now reachable, tests run versus not run, and any remaining uncertainty. Preserve useful code in core ownership; no fake fallback or compatibility-only implementation may satisfy this task. Commit after the accepted task through the authorized execution workflow. Push and merge policy follows the later execution handoff, not this planning package. Do not activate follow-on work.
