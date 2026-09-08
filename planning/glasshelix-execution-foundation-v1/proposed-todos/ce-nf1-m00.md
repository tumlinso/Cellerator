# CE-NF1-M00: Baseline and ownership accepted

Status on delivery: **planned**, not claimed or implemented.
Project: `cellerator`. First-class lane: `CE-NF1-L-M`. Workspace: isolated_merge.
Explicit prerequisites: `CE-NF1-A04`. Earlier entries in the lane queue also apply.

## Objective and implementation direction

Integrate the prerequisites onto the designated integration branch, rebuild the real linked path and verify the checkpoint contract. Accept source disposition, tool binding, resource policy and preserved-work boundaries.

## Acceptance and negative control

1. Accept source disposition, tool binding, resource policy and preserved-work boundaries.
2. Publish exact source, interface and evidence hashes; never reach a checkpoint on a declaration-only, skipped, stale or mock result.

This task is complete only when the relevant implementation or review is available to dependent tasks and the evidence level is stated honestly. A lower-level test does not automatically qualify a full integrated system.

## Read before editing

Read `01_SCOPE_AND_DECISIONS.md`, `02_SCIENTIFIC_AND_NUMERICAL_CONTRACTS.md`, `04_SINGLE_CONTROLLER_AND_PARALLELISM.md`, `05_CROSS_REPOSITORY_RECEIPTS.md` and this task's rows in the machine catalogs. Then inspect the live source, tests and nearest AGENTS rules. The following are starting references, not a closed list:

- `include/Cellerator/compute/operation/operation_core_v2/schema.hh`
- `include/Cellerator/execution/program/program_v2.h`
- `src/execution/program/program_v2.cc`
- `docs/relation_update_spine_v1/capabilities.md`

## Initial exclusive write scope

- `CMakeLists.txt`
- `docs/glasshelix_execution_foundation_v1/integration`
- `docs/glasshelix_execution_foundation_v1/records/ce-nf1-m00.json`

These paths coordinate work, not constrain the scientific representation. A needed adjacent owner path may be transferred or added by the controller through authoritative scope operations before edits. Do not clone Cellerator machinery inside GlassHelix to avoid that handoff. Preserve unrelated work and do not alter another planning package or authority tables directly.

## Required qualification

Evidence kind: **governance_review**. Required CTest names: not a runtime test task; use the review/peer verification gate below.

```sh
python3 -B planning/glasshelix-execution-foundation-v1/scripts/check_record.py --milestone CE-NF1-M00
```

Execution bindings are supplied through an external `NF1_EXECUTION_BINDINGS` file. Governance records use the exact path in `machine/acceptance_matrix.json`. No record is prefilled as passed. Test gates require real executable inventory, current source/build evidence and no skips. GPU groups require the actual resource lease and the same shared lock across both projects. Missing hardware blocks GPU qualification, not unrelated ready host tasks.

## Numerical, performance and ownership obligations

Keep scientific identity separate from layout, preserve input snapshot semantics and declare effects. Preparation may allocate bounded workspaces and maps; steady execution must not discover topology or silently canonicalize. Share only what full dependencies and generations permit. Unsupported operations return explicit status rather than a fake result. Qualification should record exact versus tolerance-based behavior and preserve nonfinite semantics. Do not force differentiability, biological naming or a specific learner into this task.

## Autonomy and handoff

Within this objective, make the best constructive choice and continue without per-choice user approval. Coordinate scope or developmental interface changes through the controller, update dependents and rerun affected tests. Commit meaningful work and push intermittently through the authorized workflow. Report changed source, actual reachable capabilities, evidence paths/hashes, numerical restrictions, integrated commit and remaining blockers. Do not mark a hypothetical path complete or start a separate biological experiment.
