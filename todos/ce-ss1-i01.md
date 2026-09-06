

<!-- todo-orchestrator:v2-managed:start -->
# CE-SS1-I01: Establish the bounded execution baseline and protected edit inventory

Task revision: `6760`; current project revision is in `todo-status.md`.

## Objective
Capture the actual pre-implementation source, available accelerator/toolchain, user-supplied example, and minimal forward/transpose regression baseline. Protect the two existing library edits.

## State
- Lifecycle: `done`
- Execution: `closed`
- Parallel policy: `parallel_safe`
- Result: `implemented`

## Next Action
Read the delivered review and decisions, and inspect only the affected execution, compiler and decomposition paths. Record precise hashes and current HEAD; do not interpret unavailable Todo health as an empty ledger.

## Ownership
- `exclusive`: `docs/semantic_spine_v1/baseline.json`
- `exclusive`: `docs/semantic_spine_v1/source_disposition.json`
- `forbidden`: `compat/legacy_sparse`
- `forbidden`: `components`
- `read`: `AGENTS.md`
- `read`: `include/Cellerator/compute/operation/operation_core.hh`
- `read`: `include/Cellerator/compute/operation/operation_core_v2/schema.hh`
- `read`: `include/Cellerator/execution/identity.hh`
- `read`: `library/cellerator.cell`
- `read`: `library/cellerator/cellerator.ceh`
- `read`: `planning/semantic-spine-v1/01_SCOPE_AND_DECISIONS.md`
- `read`: `planning/semantic-spine-v1/02_SEMANTIC_AND_NATIVE_CONTRACT.md`
- `read`: `planning/semantic-spine-v1/03_PARALLEL_LANES_AND_INTEGRATION.md`
- `read`: `planning/semantic-spine-v1/04_VALIDATION_AND_DEMO.md`
- `read`: `tests/math_core/feature_major_small_n_candidate_test.cu`
- `read`: `tests/math_core/transpose_backward_candidate_test.cu`
- `read`: `tests/planner_targets.cmake`

## Dependencies
_None._
<!-- todo-orchestrator:v2-managed:end -->
