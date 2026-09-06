<!-- todo-orchestrator:v2-managed:start -->
# CE-SS1-F02: Connect the real parser/Sema relation slice

Task revision: `6758`; current project revision is in `todo-status.md`.

## Objective
Demonstrate existing source parsing and semantic construction reach the shared descriptor for a bounded source fixture.

## State
- Lifecycle: `planned`
- Execution: `ready`
- Parallel policy: `parallel_safe`
- Result: `-`

## Next Action
Use existing parser/AST/Sema facilities for a small forward expression and transpose operation. Bind named domains, relations, state and exact identities through the existing environment or a narrowly defined binding adapter.

## Ownership
- `exclusive`: `include/Cellerator/compiler/sema/relation_spine_bridge.hh`
- `exclusive`: `src/compiler/sema/relation_spine_bridge.cc`
- `exclusive`: `tests/semantic_spine/compiler/source_slice_test.cc`
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
- `read`: `src/compiler/frontend/parser/expose_parser_library_and_parse_tree_dump_apis.cc`
- `read`: `src/compiler/frontend/parser/parse_non_relation_operation_families.cc`

## Dependencies
- `task`: `CE-SS1-F01`
<!-- todo-orchestrator:v2-managed:end -->
