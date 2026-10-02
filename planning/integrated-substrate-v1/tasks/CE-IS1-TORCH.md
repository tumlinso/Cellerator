# CE-IS1-TORCH — Expose native mathematics through the existing framework adapter

## Outcome

Extend CelleraTorch rather than creating another learning stack; preserve canonical masters and valid saved-primal semantics.

## Implementation direction

Repair current predecessor adapters in place or replace them with verified consumers. Framework-generated graphs need explicit source/value/tape identity, not a second owner of native coefficients.

## Dependencies and ownership

Local predecessors: CE-IS1-MERGE-B.

Writable roots: `components/CelleraTorch`, `tests/substrate/torch`. Other source may be inspected, not modified without reconciled scope. Root build/registry/header changes outside this scope are returned as proposed patches to the integration owner.

## Acceptance

- Real native forward/input-VJP/parameter-VJP and requested JVP paths are reachable through supported framework calls.
- No optimizer step hides in backward; parameter update/publication remains explicit.
- Shape, dtype, device, alias and stale-tape errors are tested; core CE still builds without Torch.

Bind focused actual test/consumer commands through the native task lifecycle before completion. Report source commit, changed paths, true capabilities, unsupported cases, evidence and requested shared changes. `no_change_required` is valid only for a fully inspected, already integrated outcome with current checks—not for an unattempted mechanism. Reference or mock checks cannot stand in for native/GPU/scientific acceptance.

## Integration handoff

Return an accepted-base commit/patch and its exact base, plus any generated artifacts required by downstream work. A done task does not copy a worktree. The root must integrate the patch into the designated lineage, then create/reconcile dependent worktrees from that commit. If this task itself consumes an independent leaf, publish/merge that prerequisite patch before dispatch rather than relying only on its completion flag.

Preserve ordered arguments, output ownership, support universes, numerical policy and current lifetime/generation rules. Use configured bounded subagents; do not create a nested planner hierarchy or use local observer models as implementation workers. See `docs/07_PARALLEL_EXECUTION.md` for budgets and merge barriers.
