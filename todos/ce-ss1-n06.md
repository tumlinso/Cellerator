

<!-- todo-orchestrator:v2-managed:start -->
# CE-SS1-N06: Prove native reuse and hand off executable evidence

Task revision: `6837`; current project revision is in `todo-status.md`.

## Objective
Close the native lane with actual CUDA tests and a minimal, traceable execution report.

## State
- Lifecycle: `done`
- Execution: `closed`
- Parallel policy: `parallel_safe`
- Result: `implemented`

## Next Action
Build a lane-local test target against the integrated foundation and existing provider libraries; do not edit root/central build files.

## Ownership
- `exclusive`: `docs/semantic_spine_v1/native_validation.json`
- `exclusive`: `src/compute/operation/prepared_relation.cu`
- `exclusive`: `tests/semantic_spine/native`
- `forbidden`: `compat/legacy_sparse`
- `forbidden`: `components`
- `read`: `AGENTS.md`
- `read`: `include/Cellerator/compute/operation/operation_core.hh`
- `read`: `include/Cellerator/compute/operation/operation_core_v2/schema.hh`
- `read`: `include/Cellerator/execution/identity.hh`
- `read`: `planning/semantic-spine-v1/01_SCOPE_AND_DECISIONS.md`
- `read`: `planning/semantic-spine-v1/02_SEMANTIC_AND_NATIVE_CONTRACT.md`
- `read`: `planning/semantic-spine-v1/03_PARALLEL_LANES_AND_INTEGRATION.md`
- `read`: `planning/semantic-spine-v1/04_VALIDATION_AND_DEMO.md`

## Dependencies
- `task`: `CE-SS1-N05`
<!-- todo-orchestrator:v2-managed:end -->
