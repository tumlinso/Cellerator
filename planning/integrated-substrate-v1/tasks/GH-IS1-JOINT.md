# GH-IS1-JOINT — Prove the three tools can compose without losing meaning

## Outcome

Run one bounded sequence-conditioned scientific witness through installed BP and CE components.

## Implementation direction

Use the architecture, preservation inventory and actual finished predecessor source. Prefer integrating useful current implementations over rewriting them. Keep the API small, the hot path explicit and the next extension inexpensive.

## Dependencies and ownership

Local predecessors: GH-IS1-PIPELINE.

Writable roots: `examples/substrate/joint`, `tests/substrate/joint`, `planning/integrated-substrate-v1/receipts/joint`. Other source may be inspected, not modified without reconciled scope. Root build/registry/header changes outside this scope are returned as proposed patches to the integration owner.

External provider gates: BP-IS1-CLOSE, CE-IS1-INTEGRATE. Check exact source-bound receipts; a local task DAG never proves another authority finished.

## Acceptance

- External BP-CLOSE provider and CE-INTEGRATE capabilities pass live/source-bound gates.
- Exact sequence provenance, CE numerical/derivative ownership and GH scientific meaning survive end to end.
- The witness is labelled engineering interoperability, not a compulsory model or genome-scale validation.

Bind focused actual test/consumer commands through the native task lifecycle before completion. Report source commit, changed paths, true capabilities, unsupported cases, evidence and requested shared changes. `no_change_required` is valid only for a fully inspected, already integrated outcome with current checks—not for an unattempted mechanism. Reference or mock checks cannot stand in for native/GPU/scientific acceptance.

## Integration handoff

Return an accepted-base commit/patch and its exact base, plus any generated artifacts required by downstream work. A done task does not copy a worktree. The root must integrate the patch into the designated lineage, then create/reconcile dependent worktrees from that commit. If this task itself consumes an independent leaf, publish/merge that prerequisite patch before dispatch rather than relying only on its completion flag.

Preserve ordered arguments, output ownership, support universes, numerical policy and current lifetime/generation rules. Use configured bounded subagents; do not create a nested planner hierarchy or use local observer models as implementation workers. See `docs/07_PARALLEL_EXECUTION.md` for budgets and merge barriers.
