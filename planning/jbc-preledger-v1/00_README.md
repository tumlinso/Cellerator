# Joint Cellerator–CellShard compiler pre-ledger plan

This package is the detailed planning layer immediately before Todo Orchestrator ledger construction. It does **not** mutate either Todo authority, create lanes, claim work, create worktrees, or implement source changes.

## Central thesis

Cellerator and CellShard are two tightly coupled levels of one biology-native compiler:

- **Cellerator** compiles typed biological mathematics and local biological execution.
- **CellShard** compiles recurrent biological organization into reusable execution atoms, compositions, bases, superatoms, persistent partials, graph-family realizations, physical representations, global decomposition, persistence, placement, residency, transport, collection, and distributed execution.

The atomic reusable execution hierarchy is the source architecture:

```text
biological observations / relations / identities / dynamics
    → overlapping atom evidence
    → exact certified execution atoms
    → typed compositions and superatoms
    → biological execution bases
    → graph-family and partial realizations
    → schedule and topology specialization
    → resident execution
```

Generic objects, shards, schedules, caches, graphs, arenas, and transports are downstream mechanisms. They are not the compiler ontology.

## Live source snapshot used

| Field | Value |
| --- | --- |
| Cellerator commit | 4e41d2a3726ca428869a99b58bcaa2e5fc3b5b6c |
| Cellerator project UUID | 0ccaac37-dbbf-448e-a5f8-def197a70aba |
| Cellerator Todo revision | 3600 |
| CellShard commit | 9187e86c476ecc1014bc3db597cf7b1d1a04a561 |
| CellShard project UUID | a52537a5-20db-4aeb-a126-dd0128c71fda |
| CellShard Todo revision | 311 |
| CellShard location | /home/tumlinson/Cellerator/components/CellShard |
| Observed | 2026-08-31 |

Both worktrees were clean when inspected. Project Control observations across the two authorities are not globally atomic; the Todo-planning agent must revalidate both cursors.

## Catalog size

- Total proposed Todos: **415**
- Cellerator: **100**
- CellShard: **315**
- Workstreams: **34**
- Proposed interfaces: **20**
- Promotion gates: **18**
- Suggested first-class lanes: **34**

## Documents

1. `01_PROGRAM_ARCHITECTURE_AND_INVARIANTS.md`
2. `02_INTERFACES_OWNERSHIP_AND_SOURCE_MAP.md`
3. `03_CELLERATOR_ATOMIZED_IMPLEMENTATION_PLAN.md`
4. `04_CELLSHARD_ATOM_EVIDENCE_COMPILER_PLAN.md`
5. `05_PERSISTENCE_RUNTIME_TOPOLOGY_TRANSPORT_PLAN.md`
6. `06_EXACT_VALIDATION_AND_BIOLOGICAL_EVIDENCE_PLAN.md`
7. `07_EXPERIMENTAL_ALTERNATIVES_AND_PROMOTION_GATES.md`
8. `08_DEPENDENCY_INTEGRATION_AND_PARALLELISM_MAP.md`
9. `09_COMPLETE_PROPOSED_TODO_CATALOG.md`
10. `10_PACKAGE_VALIDATION_REPORT.md`
11. `proposed_todos.json` — complete machine-readable records
12. `proposed_todos.csv` — flattened task index
13. `dependency_edges.csv` — explicit proposed task-to-task edges
14. `external_dependency_receipts.csv` — proposed checkpoint/interface/receipt dependencies
15. `interface_catalog.json`
16. `plan_summary.json`

## How the next Todo Orchestrator agent should use this package

- Reinspect the live source and both authorities.
- Map each workstream to a parent/workstream record and suggested first-class lane.
- Map each catalog entry to one Todo unless live source proves two entries are already fully implemented or one entry must be split further.
- Freeze only the interfaces needed by the next fan-out.
- Implement cross-authority dependencies with mirrored version/hash receipts if the workflow has no native cross-project dependency edge.
- Preserve all negative-result dispositions.
- Do not apply CE-AMP.
- Do not merge the two Todo authorities.
