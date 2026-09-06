

<!-- todo-orchestrator:v2-managed:start -->
# CE-SS1-C04: Test the contract and publish the foundation artifact

Task revision: `6812`; current project revision is in `todo-status.md`.

## Objective
Prove the shared descriptor and provisional signatures are coherent before the parallel fan-out.

## State
- Lifecycle: `done`
- Execution: `closed`
- Parallel policy: `parallel_safe`
- Result: `implemented`

## Next Action
Add a standalone, small host contract test build and declaration compilation checks for native/compiler consumers. Keep CPU code as an oracle/validation facility.

## Ownership
- `exclusive`: `docs/semantic_spine_v1/core_validation.json`
- `exclusive`: `tests/semantic_spine/core`
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
- `task`: `CE-SS1-C03`
<!-- todo-orchestrator:v2-managed:end -->
