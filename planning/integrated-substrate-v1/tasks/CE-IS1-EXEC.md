# CE-IS1-EXEC — Make prepared resident execution the normal path

## Outcome

Consolidate runtime/program/value/tape ownership and remove repeated preparation, copies and hidden synchronization from composed use.

## Implementation direction

Preserve training_program_v2 and parameter-owner mechanisms. Static prepared sweeps are the baseline. Runtime worklists, cooperative residency and capture are optional alternatives, not required architecture. Do not add a second generic scheduler.

## Dependencies and ownership

Local predecessors: CE-IS1-ADOPT.

Writable roots: `include/Cellerator/execution`, `src/execution`, `tests/substrate/execution`. Other source may be inspected, not modified without reconciled scope. Root build/registry/header changes outside this scope are returned as proposed patches to the integration owner.

## Acceptance

- One prepared multi-operation program runs repeatedly with changing values/parameters using existing owners.
- Capacity, streams, aliasing, immutable topology, value generations and saved-primal lifetimes are visible and tested.
- Direct native calls and explicit physical-output order remain available; adaptive queues are optional.

Bind focused actual test/consumer commands through the native task lifecycle before completion. Report source commit, changed paths, true capabilities, unsupported cases, evidence and requested shared changes. `no_change_required` is valid only for a fully inspected, already integrated outcome with current checks—not for an unattempted mechanism. Reference or mock checks cannot stand in for native/GPU/scientific acceptance.

## Integration handoff

Return an accepted-base commit/patch and its exact base, plus any generated artifacts required by downstream work. A done task does not copy a worktree. The root must integrate the patch into the designated lineage, then create/reconcile dependent worktrees from that commit. If this task itself consumes an independent leaf, publish/merge that prerequisite patch before dispatch rather than relying only on its completion flag.

Preserve ordered arguments, output ownership, support universes, numerical policy and current lifetime/generation rules. Use configured bounded subagents; do not create a nested planner hierarchy or use local observer models as implementation workers. See `docs/07_PARALLEL_EXECUTION.md` for budgets and merge barriers.
