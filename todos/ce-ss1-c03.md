

<!-- todo-orchestrator:v2-managed:start -->
# CE-SS1-C03: Specify provisional native pair preparation and bindings

Task revision: `6768`; current project revision is in `todo-status.md`.

## Objective
Give parallel consumers a small native execution contract for one prepared forward/transpose pair, without freezing the eventual SDK.

## State
- Lifecycle: `done`
- Execution: `closed`
- Parallel policy: `parallel_safe`
- Result: `implemented`

## Next Action
Define a declaration-only prepared pair interface with caller stream and borrowed device inputs/outputs, structured errors, topology lifetime, explicit refresh and destruction.

## Ownership
- `exclusive`: `docs/semantic_spine_v1/native_contract.md`
- `exclusive`: `include/Cellerator/compute/operation/prepared_relation.hh`
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
- `task`: `CE-SS1-C02`
<!-- todo-orchestrator:v2-managed:end -->
