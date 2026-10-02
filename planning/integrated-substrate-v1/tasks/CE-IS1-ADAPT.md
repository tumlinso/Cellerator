# CE-IS1-ADAPT — Integrate reuse, structural rewrites and trainable regrowth

## Outcome

Make parameter-aware caches, remembered deltas and externally supplied structural changes coexist under existing generation/lifetime rules.

## Implementation direction

Prior M08/M10 and N09/N12. Share common work across worlds only with full contextual equality; preserve complete hypotheses. A promoted explicit mechanism does not imply its current value is known.

## Dependencies and ownership

Local predecessors: CE-IS1-MERGE-A.

Writable roots: `include/Cellerator/math/adaptive`, `src/math/adaptive`, `tests/substrate/adaptive`. Other source may be inspected, not modified without reconciled scope. Root build/registry/header changes outside this scope are returned as proposed patches to the integration owner.

## Acceptance

- Linear and quadratic delta paths compare against last transmitted state; changing parameters invalidates caches.
- One supplied exact refactoring preserves function/readout and one explicit approximate reduction reports discrepancy.
- Tapes, slot incarnation, state initialization and optimizer reset/migration are handled coherently.

Bind focused actual test/consumer commands through the native task lifecycle before completion. Report source commit, changed paths, true capabilities, unsupported cases, evidence and requested shared changes. `no_change_required` is valid only for a fully inspected, already integrated outcome with current checks—not for an unattempted mechanism. Reference or mock checks cannot stand in for native/GPU/scientific acceptance.

## Integration handoff

Return an accepted-base commit/patch and its exact base, plus any generated artifacts required by downstream work. A done task does not copy a worktree. The root must integrate the patch into the designated lineage, then create/reconcile dependent worktrees from that commit. If this task itself consumes an independent leaf, publish/merge that prerequisite patch before dispatch rather than relying only on its completion flag.

Preserve ordered arguments, output ownership, support universes, numerical policy and current lifetime/generation rules. Use configured bounded subagents; do not create a nested planner hierarchy or use local observer models as implementation workers. See `docs/07_PARALLEL_EXECUTION.md` for budgets and merge barriers.
