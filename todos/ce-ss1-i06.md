

<!-- todo-orchestrator:v2-managed:start -->
# CE-SS1-I06: Run the supplied biological demo and CUDA memory checks

Task revision: `6852`; current project revision is in `todo-status.md`.

## Objective
Make the exact delivered example work with the implemented provisional API and demonstrate reuse without new performance claims.

## State
- Lifecycle: `in_progress`
- Execution: `claimed`
- Parallel policy: `parallel_safe`
- Result: `-`

## Next Action
Build and run the delivered regulatory_reuse.cc in its normal CUDA mode. Adjust only coordinated API spellings or necessary setup, never weaken assertions, expected values or required execution.

## Ownership
- `exclusive`: `docs/semantic_spine_v1/demo_execution.json`
- `exclusive`: `examples/semantic_spine_v1`
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
- `task`: `CE-SS1-I05`
<!-- todo-orchestrator:v2-managed:end -->
