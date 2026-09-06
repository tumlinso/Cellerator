

<!-- todo-orchestrator:v2-managed:start -->
# CE-SS1-F04: Publish a source-origin conformance test artifact

Task revision: `6834`; current project revision is in `todo-status.md`.

## Objective
Provide a real origin-parity test route and candid language capability status.

## State
- Lifecycle: `done`
- Execution: `closed`
- Parallel policy: `parallel_safe`
- Result: `implemented`

## Next Action
Create the lane-local compile/run test harness and record actual parser, Sema and lowering symbols exercised.

## Ownership
- `exclusive`: `docs/semantic_spine_v1/compiler_validation.json`
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
- `task`: `CE-SS1-F03`
<!-- todo-orchestrator:v2-managed:end -->
