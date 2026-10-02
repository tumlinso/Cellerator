# CE-IS1-MATH — Extend the algebra with executable frontier operators

## Outcome

Build exact quadratic deltas, boundary-port solves and residual/multilevel variants through the integrated operator interface.

## Implementation direction

Reuse prior exact flows and M11 feedback machinery; do not implement a second solver stack. Exact quadratic delta includes the delta*M*delta term. Low-rank recompression and nonlinear jets are approximations unless exact structure is supplied.

## Dependencies and ownership

Local predecessors: CE-IS1-MERGE-A.

Writable roots: `include/Cellerator/math/frontier`, `src/math/frontier`, `tests/substrate/frontier`. Other source may be inspected, not modified without reconciled scope. Root build/registry/header changes outside this scope are returned as proposed patches to the integration owner.

## Acceptance

- New N04–N06 reference identities and native prototypes execute through CE, with explicit exact/model/approximation labels.
- Conditioning and solve residuals are checked; no unstable explicit inverse is introduced for a tensor-core marketing claim.
- At least two nontrivial new compositions have actual native consumers; unsupported variants remain mapped and callable references where useful.

Bind focused actual test/consumer commands through the native task lifecycle before completion. Report source commit, changed paths, true capabilities, unsupported cases, evidence and requested shared changes. `no_change_required` is valid only for a fully inspected, already integrated outcome with current checks—not for an unattempted mechanism. Reference or mock checks cannot stand in for native/GPU/scientific acceptance.

## Integration handoff

Return an accepted-base commit/patch and its exact base, plus any generated artifacts required by downstream work. A done task does not copy a worktree. The root must integrate the patch into the designated lineage, then create/reconcile dependent worktrees from that commit. If this task itself consumes an independent leaf, publish/merge that prerequisite patch before dispatch rather than relying only on its completion flag.

Preserve ordered arguments, output ownership, support universes, numerical policy and current lifetime/generation rules. Use configured bounded subagents; do not create a nested planner hierarchy or use local observer models as implementation workers. See `docs/07_PARALLEL_EXECUTION.md` for budgets and merge barriers.
