# CE-IS1-DIFF — Compile and integrate requested response computation

## Outcome

Make forward, state/parameter VJP and JVP composable first-class capabilities using saved-primal ownership.

## Implementation direction

New N07 and prior M09. Differentiate the declared numerical/model operation, not a convenient different implementation. State and parameter tangents are independently tagged. Higher-order support is optional and explicitly advertised.

## Dependencies and ownership

Local predecessors: CE-IS1-MERGE-A.

Writable roots: `include/Cellerator/math/response`, `src/math/response`, `tests/substrate/response`. Other source may be inspected, not modified without reconciled scope. Root build/registry/header changes outside this scope are returned as proposed patches to the integration owner.

## Acceptance

- Capabilities match implementations and are checked against references, finite differences and adjoint identities.
- Response panels use genuinely requested directions; nonlinear chain rules are not applied as independent primal columns.
- Primal-zero/response-live cases, repeated coefficients and cross-generation tapes are protected.

Bind focused actual test/consumer commands through the native task lifecycle before completion. Report source commit, changed paths, true capabilities, unsupported cases, evidence and requested shared changes. `no_change_required` is valid only for a fully inspected, already integrated outcome with current checks—not for an unattempted mechanism. Reference or mock checks cannot stand in for native/GPU/scientific acceptance.

## Integration handoff

Return an accepted-base commit/patch and its exact base, plus any generated artifacts required by downstream work. A done task does not copy a worktree. The root must integrate the patch into the designated lineage, then create/reconcile dependent worktrees from that commit. If this task itself consumes an independent leaf, publish/merge that prerequisite patch before dispatch rather than relying only on its completion flag.

Preserve ordered arguments, output ownership, support universes, numerical policy and current lifetime/generation rules. Use configured bounded subagents; do not create a nested planner hierarchy or use local observer models as implementation workers. See `docs/07_PARALLEL_EXECUTION.md` for budgets and merge barriers.
