

<!-- todo-orchestrator:v2-managed:start -->
# CE-SS1-I04: Integrate all four lanes and repository-local targets

Task revision: `6812`; current project revision is in `todo-status.md`.

## Objective
Join the independent native, compiler, algebra and verification artifacts without introducing a second runtime or broad build refactor.

## State
- Lifecycle: `planned`
- Execution: `ready`
- Parallel policy: `parallel_safe`
- Result: `-`

## Next Action
Integrate all lane artifacts against the same foundation. Resolve common build changes here only; substantive algorithm/semantic fixes go back to the owning lane.

## Ownership
- `exclusive`: `CMakeLists.txt`
- `exclusive`: `cmake/SemanticSpineV1.cmake`
- `exclusive`: `docs/semantic_spine_v1/integration.json`
- `exclusive`: `examples/CMakeLists.txt`
- `exclusive`: `examples/semantic_spine_v1/CMakeLists.txt`
- `exclusive`: `tests/semantic_spine/CMakeLists.txt`
- `forbidden`: `compat/legacy_sparse`
- `forbidden`: `components`
- `read`: `AGENTS.md`
- `read`: `examples/CMakeLists.txt`
- `read`: `include/Cellerator/compute/operation/operation_core.hh`
- `read`: `include/Cellerator/compute/operation/operation_core_v2/schema.hh`
- `read`: `include/Cellerator/execution/identity.hh`
- `read`: `planning/semantic-spine-v1/01_SCOPE_AND_DECISIONS.md`
- `read`: `planning/semantic-spine-v1/02_SEMANTIC_AND_NATIVE_CONTRACT.md`
- `read`: `planning/semantic-spine-v1/03_PARALLEL_LANES_AND_INTEGRATION.md`
- `read`: `planning/semantic-spine-v1/04_VALIDATION_AND_DEMO.md`
- `read`: `src/compiler/ir/semantic/implement_relation_apply_and_transpose_operations.cc`
- `read`: `src/compiler/sema/implement_operation_kind_resolution.cc`
- `read`: `src/compute/operation/prepared_relation.cu`

## Dependencies
- `task`: `CE-SS1-N06`
- `task`: `CE-SS1-F04`
- `task`: `CE-SS1-A04`
- `task`: `CE-SS1-V04`
- `task`: `CE-SS1-I03`
<!-- todo-orchestrator:v2-managed:end -->
