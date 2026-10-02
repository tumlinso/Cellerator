# CE-IS1-MERGE-B — Integrate strategies, responses and frontier math

## Outcome

Merge the second numerical wave into the established substrate, remove replaced duplicates and re-run native compositions.

## Implementation direction

Use the architecture, preservation inventory and actual finished predecessor source. Prefer integrating useful current implementations over rewriting them. Keep the API small, the hot path explicit and the next extension inexpensive.

## Dependencies and ownership

Local predecessors: CE-IS1-STRATEGIES, CE-IS1-MATH, CE-IS1-DIFF, CE-IS1-ADAPT, CE-IS1-LOWER.

Writable roots: `CMakeLists.txt`, `cmake`, `include`, `src`, `components`, `tests`, `examples`, `experiments`, `planning/integrated-substrate-v1/receipts`. Other source may be inspected, not modified without reconciled scope. Root build/registry/header changes outside this scope are returned as proposed patches to the integration owner.

## Acceptance

- The integrated program can mix appropriate strategy/provider families without losing logical meaning or gradients.
- New math, derivatives and adaptive lifetime behavior are checked together; leaves are not merely marked done.
- All source migrations have one preferred callable entry and evidence-preserving archived drivers.

Bind focused actual test/consumer commands through the native task lifecycle before completion. Report source commit, changed paths, true capabilities, unsupported cases, evidence and requested shared changes. `no_change_required` is valid only for a fully inspected, already integrated outcome with current checks—not for an unattempted mechanism. Reference or mock checks cannot stand in for native/GPU/scientific acceptance.

## Integration handoff

Return an accepted-base commit/patch and its exact base, plus any generated artifacts required by downstream work. A done task does not copy a worktree. The root must integrate the patch into the designated lineage, then create/reconcile dependent worktrees from that commit. If this task itself consumes an independent leaf, publish/merge that prerequisite patch before dispatch rather than relying only on its completion flag.

Preserve ordered arguments, output ownership, support universes, numerical policy and current lifetime/generation rules. Use configured bounded subagents; do not create a nested planner hierarchy or use local observer models as implementation workers. See `docs/07_PARALLEL_EXECUTION.md` for budgets and merge barriers.
