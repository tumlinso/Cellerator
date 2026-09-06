

<!-- todo-orchestrator:v2-managed:start -->
# CE-SS1-C02: Implement shared semantic validation and fieldwise equivalence

Task revision: `6834`; current project revision is in `todo-status.md`.

## Objective
Both native and compiler callers must use the same relation semantics checks and equality rules.

## State
- Lifecycle: `done`
- Execution: `closed`
- Parallel policy: `parallel_safe`
- Result: `implemented`

## Next Action
Implement explicit fieldwise comparison and validation; never compare padding or use pointer addresses as persistent keys.

## Ownership
- `exclusive`: `src/compute/operation/relation_semantics.cc`
- `exclusive`: `tests/semantic_spine/core/descriptor_test.cc`
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
- `task`: `CE-SS1-C01`
<!-- todo-orchestrator:v2-managed:end -->
