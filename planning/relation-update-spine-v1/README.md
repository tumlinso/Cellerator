# Relation Update Spine v1

**Planning/manual-bootstrap package and prospective demonstration. Not an implementation release.**

This package extends the proven Semantic Spine into one small mutable biological computation: N16 forward, transpose, edge-value VJP, caller delta and gradient-step updates, true readiness publication and same-topology reuse. It includes repaired/reachable WMMA with exact sparse residual/fallback, both semantic origins, and V100 acceptance. Useful existing algorithms belong in shared core; no general training engine is being introduced.

The scope has **34 leaf tasks, 35 native Todo records, eight lane definitions, 45 explicit task dependencies, five checkpoints and three checkpoint-derived barriers**. There is one common developmental interface and no cross-repository task dependency. The native plan is schema 3 and passed live read-only Project Control validation with 35 additions, no modifications, warnings, dependency errors, interface errors or scope conflicts. No Todo state was applied or activated.

A single-page reading copy is available in `REVIEW_AND_BOOTSTRAP.html`.

## Read first

`01_SCOPE_AND_DECISIONS.md` sets the goal and boundaries. `02_SEMANTIC_AND_NATIVE_CONTRACT.md`, `03_WMMA_AND_DECOMPOSITION.md` and `04_READINESS_AND_OWNERSHIP.md` resolve the principal engineering questions. `05_VALIDATION_AND_DEMO.md` and `machine/acceptance_matrix.json` define evidence. `06_PARALLEL_LANES_AND_INTEGRATION.md` defines ownership. `07_SOURCE_LEDGER.md` and `10_RESEARCH_AND_DECISION_LOG.md` separate observations from decisions and research. `08_DEFERRED_WORK.md` preserves near-term WMMA and broader long-term biological-learning work. `09_MANUAL_BOOTSTRAP.md` is the operator procedure.

The complete review is preserved under `basis/`. Every native task points to a substantive `proposed-todos/` sheet. `handoff/` contains prospective lane instructions; these do not start agents. `contracts/` contains declaration sketches only, not installed headers. The demo is at `examples/relation_update_spine_v1/` in the repository-relative overlay.

## Machine authority

Only `machine/relation-update-spine-v1.todo-plan.json` is applied by the native schema-3 front door. The rich catalog and CSV/JSON projections are planning/consistency views. `scripts/compile_plan.py --check` verifies the native projection. `scripts/validate_package.py` validates the combined dependency/lane graph, scope ownership, projections, package hashes and delivered demo. `scripts/test_package.py` exercises failure cases without touching authority. `scripts/todo_bootstrap.py` provides offline validation, fresh native preview and explicitly confirmed manual ingestion, never activation.

## Delivery status

The final plan was validated against clean Cellerator `main` at `c2830e3420ddae5c5ea92fa270cb3574fc19cd36`, Todo/workflow revision 6877. All repository reads and validation were read-only. The overlay itself was created in a separate artifact filesystem; it has not been copied into your repository.

The reference-only demo was compiled and run locally; its fixture mathematics, adjoint, finite differences and half rounding passed. The normal post-epic consumer received a declaration-only syntax check, not real linking or CUDA execution. Its required core APIs are not implemented by this package. GPU acceptance, sanitizer and performance measurements are future mandatory tasks. `evidence/` distinguishes these statuses explicitly.

To inspect now, run `python3 -B planning/relation-update-spine-v1/scripts/validate_package.py` from the extracted overlay. To ingest later, use the staged instructions in `09_MANUAL_BOOTSTRAP.md`. Do not apply the preledger with a generic compiler, manually change authority tables, or activate a run as a side effect of package installation.
