# CE-IS1-PACK — Publish a format-independent packing strategy seam

## Outcome

Separate strategy selection, semantic realization and physical lowering; wrap existing generic occupancy and Cellpack capabilities without making them mandatory.

## Implementation direction

Read evaluator.hh, packing_plan.hh and row_masked_n1_candidate.hh. The generic view and frozen image have different scope. Preserve exact occupancy vs hypothetical cost distinction. Start from the supplied Python/C++ contract witnesses, not a fresh plugin infrastructure.

## Dependencies and ownership

Local predecessors: CE-IS1-ADOPT.

Writable roots: `include/Cellerator/packing`, `src/packing/core`, `tests/substrate/packing`. Other source may be inspected, not modified without reconciled scope. Root build/registry/header changes outside this scope are returned as proposed patches to the integration owner.

## Acceptance

- Identity, supplied-plan and actual Cellpack adapter paths conform to one cold strategy contract.
- Independent source/destination/state/op orders and contribution ownership are representable; unsupported projections fail clearly.
- The current CPK1 N=1 candidate stays narrow and usable; no hidden universal conversion through Cellpack.

Bind focused actual test/consumer commands through the native task lifecycle before completion. Report source commit, changed paths, true capabilities, unsupported cases, evidence and requested shared changes. `no_change_required` is valid only for a fully inspected, already integrated outcome with current checks—not for an unattempted mechanism. Reference or mock checks cannot stand in for native/GPU/scientific acceptance.

## Integration handoff

Return an accepted-base commit/patch and its exact base, plus any generated artifacts required by downstream work. A done task does not copy a worktree. The root must integrate the patch into the designated lineage, then create/reconcile dependent worktrees from that commit. If this task itself consumes an independent leaf, publish/merge that prerequisite patch before dispatch rather than relying only on its completion flag.

Preserve ordered arguments, output ownership, support universes, numerical policy and current lifetime/generation rules. Use configured bounded subagents; do not create a nested planner hierarchy or use local observer models as implementation workers. See `docs/07_PARALLEL_EXECUTION.md` for budgets and merge barriers.
