

<!-- todo-orchestrator:v2-managed:start -->
# CE-SS1-N05: Execute transpose with shared topology and value authority

Task revision: `6834`; current project revision is in `todo-status.md`.

## Objective
Prove reverse application uses the same biological relation and current values, not a second independent semantic model.

## State
- Lifecycle: `done`
- Execution: `closed`
- Parallel policy: `parallel_safe`
- Result: `implemented`

## Next Action
Bind destination-axis input and source-axis output to CTP1; do not rewrite the relation IDs to mean the reverse graph.

## Ownership
- `exclusive`: `src/compute/operation/prepared_relation.cu`
- `exclusive`: `tests/semantic_spine/native/transpose_test.cu`
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
- `task`: `CE-SS1-N04`
<!-- todo-orchestrator:v2-managed:end -->
