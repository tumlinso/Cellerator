# CE-IS1-LOWER — Search and integrate useful Volta-native realizations

## Outcome

Add bounded typed rewrite/specialization and instruction/layout alternatives through the existing compiler/provider seam.

## Implementation direction

New N10–N11. Full Volta palette allowed; branch-free dogma is not. A small enumerator or local rewrite set is enough—do not require a new universal e-graph compiler. Preserve exploration outputs even when they lose.

## Dependencies and ownership

Local predecessors: CE-IS1-MERGE-A.

Writable roots: `include/Cellerator/compiler/substrate`, `src/compiler/substrate`, `src/compute/architecture/providers/nvidia/sm70/substrate`, `tests/substrate/lowering`. Other source may be inspected, not modified without reconciled scope. Root build/registry/header changes outside this scope are returned as proposed patches to the integration owner.

## Acceptance

- SM70 capability filters reject later-architecture instructions; emitted instructions/register costs are inspected.
- At least one fused or specialized composition competes against a direct/vendor baseline with the same declared semantics.
- M04/M12/M13 and BP numerical machine ideas remain explicit alternatives with consistent numerical/derivative policies.

Bind focused actual test/consumer commands through the native task lifecycle before completion. Report source commit, changed paths, true capabilities, unsupported cases, evidence and requested shared changes. `no_change_required` is valid only for a fully inspected, already integrated outcome with current checks—not for an unattempted mechanism. Reference or mock checks cannot stand in for native/GPU/scientific acceptance.

## Integration handoff

Return an accepted-base commit/patch and its exact base, plus any generated artifacts required by downstream work. A done task does not copy a worktree. The root must integrate the patch into the designated lineage, then create/reconcile dependent worktrees from that commit. If this task itself consumes an independent leaf, publish/merge that prerequisite patch before dispatch rather than relying only on its completion flag.

Preserve ordered arguments, output ownership, support universes, numerical policy and current lifetime/generation rules. Use configured bounded subagents; do not create a nested planner hierarchy or use local observer models as implementation workers. See `docs/07_PARALLEL_EXECUTION.md` for budgets and merge barriers.
