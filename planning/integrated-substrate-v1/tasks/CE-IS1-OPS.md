# CE-IS1-OPS — Integrate the current matrix and process mathematics

## Outcome

Turn completed matrix-patch, nonlinear polynomial, port/process and indexed-mechanism work into callable reusable native operations.

## Implementation direction

Retain ordered slots and output effects from indexed incidence. Plain LXR is a restricted model family, not an equivalent arbitrary sparse-matmul lowering. Actual SM70 implementations integrate through existing providers; shared registry edits go to MERGE-A.

## Dependencies and ownership

Local predecessors: CE-IS1-ADOPT.

Writable roots: `include/Cellerator/math/matrix`, `src/math/matrix`, `include/Cellerator/math/process`, `src/math/process`, `tests/substrate/operators`. Other source may be inspected, not modified without reconciled scope. Root build/registry/header changes outside this scope are returned as proposed patches to the integration owner.

## Acceptance

- Existing M01–M06 functions are represented or explicitly mapped to supported/experimental alternatives, not reimplemented as a new runtime.
- Each operation declares precise axes, numerical policy and real derivative capabilities; repeated argument/parameter multiplicity survives.
- One installed/native consumer executes both a scalar patch and a process/port computation.

Bind focused actual test/consumer commands through the native task lifecycle before completion. Report source commit, changed paths, true capabilities, unsupported cases, evidence and requested shared changes. `no_change_required` is valid only for a fully inspected, already integrated outcome with current checks—not for an unattempted mechanism. Reference or mock checks cannot stand in for native/GPU/scientific acceptance.

## Integration handoff

Return an accepted-base commit/patch and its exact base, plus any generated artifacts required by downstream work. A done task does not copy a worktree. The root must integrate the patch into the designated lineage, then create/reconcile dependent worktrees from that commit. If this task itself consumes an independent leaf, publish/merge that prerequisite patch before dispatch rather than relying only on its completion flag.

Preserve ordered arguments, output ownership, support universes, numerical policy and current lifetime/generation rules. Use configured bounded subagents; do not create a nested planner hierarchy or use local observer models as implementation workers. See `docs/07_PARALLEL_EXECUTION.md` for budgets and merge barriers.
