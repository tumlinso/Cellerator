# CE-IS1-EFFECTS — Integrate composable effects, residuals and port algebra

## Outcome

Unify general numerical providers from both moonshot campaigns and add precise reusable effect carriers.

## Implementation direction

Use new cards N01–N03. Baseplane keeps source/sequence adapters. A compact carrier is selected only when it pays for its size; neither arbitrary nonlinear functions nor dense low-rank products are assumed closed under composition.

## Dependencies and ownership

Local predecessors: CE-IS1-ADOPT.

Writable roots: `include/Cellerator/math/effects`, `src/math/effects`, `tests/substrate/effects`. Other source may be inspected, not modified without reconciled scope. Root build/registry/header changes outside this scope are returned as proposed patches to the integration owner.

## Acceptance

- DFA/counted, monomial/block-affine, lifting/residual and finite-relation operators have a single CE owner.
- New accumulated-observable affine and hybrid discrete/continuous compositions match direct application in reference tests.
- Sequential order, overflow, floating association and expansion-point limits are explicit.

Bind focused actual test/consumer commands through the native task lifecycle before completion. Report source commit, changed paths, true capabilities, unsupported cases, evidence and requested shared changes. `no_change_required` is valid only for a fully inspected, already integrated outcome with current checks—not for an unattempted mechanism. Reference or mock checks cannot stand in for native/GPU/scientific acceptance.

## Integration handoff

Return an accepted-base commit/patch and its exact base, plus any generated artifacts required by downstream work. A done task does not copy a worktree. The root must integrate the patch into the designated lineage, then create/reconcile dependent worktrees from that commit. If this task itself consumes an independent leaf, publish/merge that prerequisite patch before dispatch rather than relying only on its completion flag.

Preserve ordered arguments, output ownership, support universes, numerical policy and current lifetime/generation rules. Use configured bounded subagents; do not create a nested planner hierarchy or use local observer models as implementation workers. See `docs/07_PARALLEL_EXECUTION.md` for budgets and merge barriers.
