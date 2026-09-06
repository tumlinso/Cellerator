<!-- todo-orchestrator:v2-managed:start -->
# CE-SS1-V01: Create an independent logical-edge oracle and adversarial fixtures

Task revision: `6758`; current project revision is in `todo-status.md`.

## Objective
Make correctness independent of physical packing and compiler lowering.

## State
- Lifecycle: `planned`
- Execution: `ready`
- Parallel policy: `parallel_safe`
- Result: `-`

## Next Action
Use small host double accumulation over logical edges for oracle values. Retain declared input quantization; compare GPU output to the same stored inputs, not an unrelated higher-precision problem.

## Ownership
- `exclusive`: `tests/semantic_spine/verification/fixtures.hh`
- `exclusive`: `tests/semantic_spine/verification/oracle.hh`
- `exclusive`: `tests/semantic_spine/verification/oracle_test.cc`
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
- `task`: `CE-SS1-I02`
<!-- todo-orchestrator:v2-managed:end -->
