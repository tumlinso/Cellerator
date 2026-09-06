<!-- todo-orchestrator:v2-managed:start -->
# CE-SS1-V04: Deliver validation harness and adversarial controls

Task revision: `6758`; current project revision is in `todo-status.md`.

## Objective
Give integration a compact test harness and prove it detects wrong output and missing hardware.

## State
- Lifecycle: `planned`
- Execution: `ready`
- Parallel policy: `parallel_safe`
- Result: `-`

## Next Action
Compile/run independent oracle and metadata tests locally. Mark integrated probes as pending linkage rather than passed.

## Ownership
- `exclusive`: `docs/semantic_spine_v1/verification_handoff.json`
- `exclusive`: `tests/semantic_spine/verification`
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
- `task`: `CE-SS1-V03`
<!-- todo-orchestrator:v2-managed:end -->
