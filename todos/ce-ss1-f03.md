

<!-- todo-orchestrator:v2-managed:start -->
# CE-SS1-F03: Preserve effects and diagnostics without losing equivalence

Task revision: `6824`; current project revision is in `todo-status.md`.

## Objective
Prevent compiler wrappers from silently weakening the native contract.

## State
- Lifecycle: `done`
- Execution: `closed`
- Parallel policy: `parallel_safe`
- Result: `implemented`

## Next Action
Check high identity bits, axis orientation, logical edge order, empty shapes, separate arithmetic types and unsupported output/alias modes.

## Ownership
- `exclusive`: `src/compiler/ir/semantic/implement_relation_apply_and_transpose_operations.cc`
- `exclusive`: `src/compiler/sema/relation_spine_bridge.cc`
- `exclusive`: `tests/compiler/semantic_ir/implement_relation_apply_and_transpose_operations_test.cc`
- `exclusive`: `tests/semantic_spine/compiler/diagnostic_test.cc`
- `exclusive`: `tests/semantic_spine/compiler/lowering_test.cc`
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
- `task`: `CE-SS1-F02`
<!-- todo-orchestrator:v2-managed:end -->
