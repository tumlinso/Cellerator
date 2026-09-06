<!-- todo-orchestrator:v2-managed:start -->
# CE-SS1-A02: Make primitive, composition and effect classifications explicit

Task revision: `6758`; current project revision is in `todo-status.md`.

## Objective
Prevent high-level constructs from being mistaken for the representative primitive opcode used by an older mapping.

## State
- Lifecycle: `planned`
- Execution: `ready`
- Parallel policy: `parallel_safe`
- Result: `-`

## Next Action
Replace the meaningful ambiguity in operation_kind_resolution with a tagged primitive/composition/effect result or equivalent non-ambiguous representation.

## Ownership
- `exclusive`: `include/Cellerator/compiler/sema/implement_operation_kind_resolution_v1.hh`
- `exclusive`: `src/compiler/sema/implement_operation_kind_resolution.cc`
- `exclusive`: `tests/semantic_spine/algebra/classification_test.cc`
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
- `task`: `CE-SS1-A01`
<!-- todo-orchestrator:v2-managed:end -->
