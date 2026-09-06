

<!-- todo-orchestrator:v2-managed:start -->
# CE-SS1-F05: Repair numeric contract parity in retained relation lowering

Task revision: `6839`; current project revision is in `todo-status.md`.

## Objective
Repair the audited divergence between canonical relation arithmetic and retained operation/algebra transport without expanding the executable portfolio.

## State
- Lifecycle: `done`
- Execution: `closed`
- Parallel policy: `parallel_safe`
- Result: `implemented`

## Next Action
Add failing frontend numeric parity regressions, derive retained numeric transport from canonical arithmetic, and fail closed for unrepresentable restricted arithmetic policies.

## Ownership
- `exclusive`: `docs/semantic_spine_v1/compiler_validation.json`
- `exclusive`: `include/Cellerator/compiler/ir/semantic/implement_relation_apply_and_transpose_operations_v1.hh`
- `exclusive`: `src/compiler/ir/semantic/implement_relation_apply_and_transpose_operations.cc`
- `exclusive`: `tests/semantic_spine/compiler`
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
- `task`: `CE-SS1-F04`
<!-- todo-orchestrator:v2-managed:end -->
