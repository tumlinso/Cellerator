# CE-IS1-STATE — Unify structured state, support and readout objects

## Outcome

Make scalar patches and actor-private/ragged hidden state usable native objects with explicit semantics and owner-compatible views.

## Implementation direction

Map prior M01/M05/M07/M08 and STATE outputs here. Do not force uniform hidden width or globally named latent columns. Storage views are optional projections; no compulsory per-step transpose/canonicalization.

## Dependencies and ownership

Local predecessors: CE-IS1-ADOPT.

Writable roots: `include/Cellerator/state`, `src/state`, `tests/substrate/state`. Other source may be inspected, not modified without reconciled scope. Root build/registry/header changes outside this scope are returned as proposed patches to the integration owner.

## Acceptance

- Scalar-state and actor-private examples preserve logical identity, readout meaning and both useful algebraic axes.
- Supports carry universe/type/version; duplicate IDs, physical replicas and stale slot incarnations are tested.
- Existing memory/generation owners are reused; no second allocator or canonical parameter owner.

Bind focused actual test/consumer commands through the native task lifecycle before completion. Report source commit, changed paths, true capabilities, unsupported cases, evidence and requested shared changes. `no_change_required` is valid only for a fully inspected, already integrated outcome with current checks—not for an unattempted mechanism. Reference or mock checks cannot stand in for native/GPU/scientific acceptance.

## Integration handoff

Return an accepted-base commit/patch and its exact base, plus any generated artifacts required by downstream work. A done task does not copy a worktree. The root must integrate the patch into the designated lineage, then create/reconcile dependent worktrees from that commit. If this task itself consumes an independent leaf, publish/merge that prerequisite patch before dispatch rather than relying only on its completion flag.

Preserve ordered arguments, output ownership, support universes, numerical policy and current lifetime/generation rules. Use configured bounded subagents; do not create a nested planner hierarchy or use local observer models as implementation workers. See `docs/07_PARALLEL_EXECUTION.md` for budgets and merge barriers.
