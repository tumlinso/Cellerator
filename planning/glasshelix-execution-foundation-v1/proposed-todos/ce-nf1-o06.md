# CE-NF1-O06: Qualify bounded read-only replay where supported

Status on delivery: **planned**, not claimed or implemented.
Project: `cellerator`. First-class lane: `CE-NF1-L-O`. Workspace: isolated_merge.
Explicit prerequisites: `CE-NF1-O05`. Earlier entries in the lane queue also apply.

## Objective and implementation direction

Add or qualify repeated fixed-program replay only for capture-safe operations after ordinary stream execution is correct.

## Acceptance and negative control

1. Supported replay produces current input-bound results and measurable evidence; unsupported mutable capture remains explicit.
2. No host generation counter is falsely advanced by graph replay; capture is not required for unrelated correctness.

This task is complete only when the relevant implementation or review is available to dependent tasks and the evidence level is stated honestly. A lower-level test does not automatically qualify a full integrated system.

## Read before editing

Read `01_SCOPE_AND_DECISIONS.md`, `02_SCIENTIFIC_AND_NUMERICAL_CONTRACTS.md`, `04_SINGLE_CONTROLLER_AND_PARALLELISM.md`, `05_CROSS_REPOSITORY_RECEIPTS.md` and this task's rows in the machine catalogs. Then inspect the live source, tests and nearest AGENTS rules. The following are starting references, not a closed list:

- `docs/relation_update_spine_v1/evidence/performance_analysis.md`
- `include/Cellerator/planner/end_to_end_planner.hh`

## Initial exclusive write scope

- `include/Cellerator/compute/operation/native_foundation_optimization`
- `src/compute/operation/native_foundation_optimization`
- `bench/native_foundation`
- `docs/glasshelix_execution_foundation_v1/performance`

These paths coordinate work, not constrain the scientific representation. A needed adjacent owner path may be transferred or added by the controller through authoritative scope operations before edits. Do not clone Cellerator machinery inside GlassHelix to avoid that handoff. Preserve unrelated work and do not alter another planning package or authority tables directly.

## Required qualification

Evidence kind: **executed_cuda_test**. Required CTest names: `ce_nf1_o06`.

```sh
python3 -B planning/glasshelix-execution-foundation-v1/scripts/run_gate.py --group o06
```

Execution bindings are supplied through an external `NF1_EXECUTION_BINDINGS` file. Governance records use the exact path in `machine/acceptance_matrix.json`. No record is prefilled as passed. Test gates require real executable inventory, current source/build evidence and no skips. GPU groups require the actual resource lease and the same shared lock across both projects. Missing hardware blocks GPU qualification, not unrelated ready host tasks.

## Numerical, performance and ownership obligations

Keep scientific identity separate from layout, preserve input snapshot semantics and declare effects. Preparation may allocate bounded workspaces and maps; steady execution must not discover topology or silently canonicalize. Share only what full dependencies and generations permit. Unsupported operations return explicit status rather than a fake result. Qualification should record exact versus tolerance-based behavior and preserve nonfinite semantics. Do not force differentiability, biological naming or a specific learner into this task.

## Autonomy and handoff

Within this objective, make the best constructive choice and continue without per-choice user approval. Coordinate scope or developmental interface changes through the controller, update dependents and rerun affected tests. Commit meaningful work and push intermittently through the authorized workflow. Report changed source, actual reachable capabilities, evidence paths/hashes, numerical restrictions, integrated commit and remaining blockers. Do not mark a hypothetical path complete or start a separate biological experiment.
