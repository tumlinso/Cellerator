# Cellerator ownership of basis selection

Todo `CE-CCP1-A03-005` moves the JBC basis search at CellShard
`b9749ad3e5146a04f847533d8c6f1a54146aed20` into the Cellerator planner.
The reviewed family includes input/budget, baseline, greedy, exact oracle,
facility location, set cover, overlap, multi-basis, Pareto, refinement,
split/merge, swap, promotion, membership, utility and manifest contracts.

Basis inputs are representative-profile identities and generations, workload
family identities, exact required atoms and structure epochs. Outputs are
portable Cellerator rulesets with a selected basis, a redundant-bases result,
or an explicit valid no-basis outcome. Complete cost includes build, storage,
materialization, execution, canonicalization and invalidation; selection is not
permitted to optimize only stored bytes or a microkernel.

Cellerator owns search, exact/heuristic comparison, cost reasoning, promotion
and portable basis identity. CellShard retains concrete materialization,
persistence, placement and storage cache policy. The deliberate namespace
change preserves current algorithms and negative outcomes without making a
stored basis manifest the semantic authority.

Gate `CE-CCP1-A03-005-GATE` proves profile-to-ruleset traceability, accepts an
explicit no-basis result without fabricating an identity, rejects stale or
mismatched profile generations, and checks all complete-cost terms contribute.
