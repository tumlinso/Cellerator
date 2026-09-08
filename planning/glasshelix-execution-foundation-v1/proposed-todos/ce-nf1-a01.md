# CE-NF1-A01: Revalidate source and both planning precedents

Status on delivery: **planned**, not claimed or implemented.
Project: `cellerator`. First-class lane: `CE-NF1-L-A`. Workspace: isolated_merge.
Explicit prerequisites: none. Earlier entries in the lane queue also apply.

## Objective and implementation direction

Inspect current AGENTS, scope, operation-core v2, JBC transition map, SS1 and RU1 source/tests; classify implemented, unlinked, reference-only and absent behavior before changing it.

## Acceptance and negative control

1. Record actual HEAD, worktrees, dirty files, task authority and callable target map.
2. Existing dirty work and unrelated active runs remain unchanged; old completion labels never substitute for code inspection.

This task is complete only when the relevant implementation or review is available to dependent tasks and the evidence level is stated honestly. A lower-level test does not automatically qualify a full integrated system.

## Read before editing

Read `01_SCOPE_AND_DECISIONS.md`, `02_SCIENTIFIC_AND_NUMERICAL_CONTRACTS.md`, `04_SINGLE_CONTROLLER_AND_PARALLELISM.md`, `05_CROSS_REPOSITORY_RECEIPTS.md` and this task's rows in the machine catalogs. Then inspect the live source, tests and nearest AGENTS rules. The following are starting references, not a closed list:

- `AGENTS.md`
- `scope.md`
- `docs/JBC/CE_JBC_B02_TRANSITION_MAP.md`
- `docs/relation_update_spine_v1/capabilities.md`

## Initial exclusive write scope

- `docs/glasshelix_execution_foundation_v1/baseline`
- `docs/glasshelix_execution_foundation_v1/ownership`
- `docs/glasshelix_execution_foundation_v1/records/ce-nf1-a01.json`

These paths coordinate work, not constrain the scientific representation. A needed adjacent owner path may be transferred or added by the controller through authoritative scope operations before edits. Do not clone Cellerator machinery inside GlassHelix to avoid that handoff. Preserve unrelated work and do not alter another planning package or authority tables directly.

## Required qualification

Evidence kind: **governance_review**. Required CTest names: not a runtime test task; use the review/peer verification gate below.

```sh
python3 -B planning/glasshelix-execution-foundation-v1/scripts/check_record.py --milestone CE-NF1-A01
```

Execution bindings are supplied through an external `NF1_EXECUTION_BINDINGS` file. Governance records use the exact path in `machine/acceptance_matrix.json`. No record is prefilled as passed. Test gates require real executable inventory, current source/build evidence and no skips. GPU groups require the actual resource lease and the same shared lock across both projects. Missing hardware blocks GPU qualification, not unrelated ready host tasks.

## Numerical, performance and ownership obligations

Keep scientific identity separate from layout, preserve input snapshot semantics and declare effects. Preparation may allocate bounded workspaces and maps; steady execution must not discover topology or silently canonicalize. Share only what full dependencies and generations permit. Unsupported operations return explicit status rather than a fake result. Qualification should record exact versus tolerance-based behavior and preserve nonfinite semantics. Do not force differentiability, biological naming or a specific learner into this task.

## Autonomy and handoff

Within this objective, make the best constructive choice and continue without per-choice user approval. Coordinate scope or developmental interface changes through the controller, update dependents and rerun affected tests. Commit meaningful work and push intermittently through the authorized workflow. Report changed source, actual reachable capabilities, evidence paths/hashes, numerical restrictions, integrated commit and remaining blockers. Do not mark a hypothetical path complete or start a separate biological experiment.
