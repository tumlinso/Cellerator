<!-- todo-orchestrator:v2-managed:start -->
# CE-SS1-F01: Lower existing relation semantic IR through the shared contract

Task revision: `6758`; current project revision is in `todo-status.md`.

## Objective
Replace the selected relation-family duplicate semantic interpretation with the canonical descriptor.

## State
- Lifecycle: `planned`
- Execution: `ready`
- Parallel policy: `parallel_safe`
- Result: `-`

## Next Action
Adapt the existing relation_apply_operation_ir_v1 lowering, not a new demo-only parser. Preserve source locations and symbol identity separately from mathematical equality.

## Ownership
- `exclusive`: `include/Cellerator/compiler/ir/semantic/implement_relation_apply_and_transpose_operations_v1.hh`
- `exclusive`: `src/compiler/ir/semantic/implement_relation_apply_and_transpose_operations.cc`
- `exclusive`: `tests/semantic_spine/compiler/lowering_test.cc`
- `forbidden`: `compat/legacy_sparse`
- `forbidden`: `components`
- `read`: `AGENTS.md`
- `read`: `include/Cellerator/compiler/ir/semantic/implement_relation_apply_and_transpose_operations_v1.hh`
- `read`: `include/Cellerator/compute/operation/operation_core.hh`
- `read`: `include/Cellerator/compute/operation/operation_core_v2/schema.hh`
- `read`: `include/Cellerator/execution/identity.hh`
- `read`: `planning/semantic-spine-v1/01_SCOPE_AND_DECISIONS.md`
- `read`: `planning/semantic-spine-v1/02_SEMANTIC_AND_NATIVE_CONTRACT.md`
- `read`: `planning/semantic-spine-v1/03_PARALLEL_LANES_AND_INTEGRATION.md`
- `read`: `planning/semantic-spine-v1/04_VALIDATION_AND_DEMO.md`
- `read`: `src/compiler/ir/semantic/implement_relation_apply_and_transpose_operations.cc`
- `read`: `src/compiler/sema/implement_operation_kind_resolution.cc`

## Dependencies
- `task`: `CE-SS1-I02`
<!-- todo-orchestrator:v2-managed:end -->
