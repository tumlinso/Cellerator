

<!-- todo-orchestrator:v2-managed:start -->
# CE-SS1-N03: Refresh device values without rebuilding topology

Task revision: `6814`; current project revision is in `todo-status.md`.

## Objective
Publish changing values as stream-ordered generations while reusing the prepared structure.

## State
- Lifecycle: `done`
- Execution: `closed`
- Parallel policy: `parallel_safe`
- Result: `implemented`

## Next Action
Bind device f16 values with structure ID/epoch, logical edge order, generation and count. Use a device gather/packing path when mapping to persistent value positions.

## Ownership
- `exclusive`: `src/compute/operation/prepared_relation.cu`
- `exclusive`: `tests/semantic_spine/native/generation_test.cu`
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
- `read`: `src/runtime/value_readiness.cu`

## Dependencies
- `task`: `CE-SS1-N02`
<!-- todo-orchestrator:v2-managed:end -->
