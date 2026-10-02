# CE-IS1-STRATEGIES — Add genuinely different packing and repair strategies

## Outcome

Implement alternative geometry/operation packers and their amortized-cost selection without coupling them to Cellpack.

## Implementation direction

New card N08. Include no-pack, Cellpack and direct vector controls. The deferred CUDA occupancy evaluator is investigated only if preparation measurements justify it; record its disposition regardless.

## Dependencies and ownership

Local predecessors: CE-IS1-MERGE-A.

Writable roots: `src/packing/strategies`, `include/Cellerator/packing/strategies`, `tests/substrate/strategies`, `bench/substrate/strategies`. Other source may be inspected, not modified without reconciled scope. Root build/registry/header changes outside this scope are returned as proposed patches to the integration owner.

## Acceptance

- At least one genuinely different packer plus a caller-defined strategy executes the same supported operation as the baseline.
- Compare two-sided relation geometry, opcode/shared-load cohorts and mixed dense/sparse candidates where applicable.
- Incremental repair and hysteresis have explicit migration cost and do not silently drop work or derivatives.

Bind focused actual test/consumer commands through the native task lifecycle before completion. Report source commit, changed paths, true capabilities, unsupported cases, evidence and requested shared changes. `no_change_required` is valid only for a fully inspected, already integrated outcome with current checks—not for an unattempted mechanism. Reference or mock checks cannot stand in for native/GPU/scientific acceptance.

## Integration handoff

Return an accepted-base commit/patch and its exact base, plus any generated artifacts required by downstream work. A done task does not copy a worktree. The root must integrate the patch into the designated lineage, then create/reconcile dependent worktrees from that commit. If this task itself consumes an independent leaf, publish/merge that prerequisite patch before dispatch rather than relying only on its completion flag.

Preserve ordered arguments, output ownership, support universes, numerical policy and current lifetime/generation rules. Use configured bounded subagents; do not create a nested planner hierarchy or use local observer models as implementation workers. See `docs/07_PARALLEL_EXECUTION.md` for budgets and merge barriers.
