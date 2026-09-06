

<!-- todo-orchestrator:v2-managed:start -->
# CE-SS1-N04: Execute canonical forward application on sm70

Task revision: `6819`; current project revision is in `todo-status.md`.

## Objective
Make canonical forward requests reach the existing device candidate through the prepared pair.

## State
- Lifecycle: `done`
- Execution: `closed`
- Parallel policy: `parallel_safe`
- Result: `implemented`

## Next Action
Bind typed input/output axes and counts, validate arithmetic/output semantics, and invoke the bound forward candidate on the caller stream.

## Ownership
- `exclusive`: `src/compute/operation/prepared_relation.cu`
- `exclusive`: `tests/semantic_spine/native/forward_test.cu`
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
- `task`: `CE-SS1-N03`
<!-- todo-orchestrator:v2-managed:end -->
