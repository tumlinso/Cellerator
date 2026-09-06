

<!-- todo-orchestrator:v2-managed:start -->
# CE-SS1-I02: Integrate the foundation and release parallel lanes

Task revision: `6771`; current project revision is in `todo-status.md`.

## Objective
Make the tested canonical contract available in one shared source base and publish its internal interface.

## State
- Lifecycle: `in_progress`
- Execution: `claimed`
- Parallel policy: `parallel_safe`
- Result: `-`

## Next Action
Integrate C04, add only the opt-in build wiring for the semantic contract, and verify its leaf tests.

## Ownership
- `exclusive`: `CMakeLists.txt`
- `exclusive`: `cmake/SemanticSpineV1.cmake`
- `exclusive`: `docs/semantic_spine_v1/foundation.json`
- `forbidden`: `compat/legacy_sparse`
- `forbidden`: `components`
- `read`: `AGENTS.md`
- `read`: `include/Cellerator/compute/operation/operation_core.hh`
- `read`: `include/Cellerator/compute/operation/operation_core_v2/schema.hh`
- `read`: `include/Cellerator/compute/operation/prepared_relation.hh`
- `read`: `include/Cellerator/compute/operation/relation_semantics.hh`
- `read`: `include/Cellerator/execution/identity.hh`
- `read`: `planning/semantic-spine-v1/01_SCOPE_AND_DECISIONS.md`
- `read`: `planning/semantic-spine-v1/02_SEMANTIC_AND_NATIVE_CONTRACT.md`
- `read`: `planning/semantic-spine-v1/03_PARALLEL_LANES_AND_INTEGRATION.md`
- `read`: `planning/semantic-spine-v1/04_VALIDATION_AND_DEMO.md`
- `read`: `src/compute/operation/relation_semantics.cc`

## Dependencies
- `task`: `CE-SS1-C04`
<!-- todo-orchestrator:v2-managed:end -->
