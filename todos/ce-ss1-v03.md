<!-- todo-orchestrator:v2-managed:start -->
# CE-SS1-V03: Specify integrated origin and GPU acceptance probes

Task revision: `6758`; current project revision is in `todo-status.md`.

## Objective
Prepare cross-origin and hardware probes against the published interfaces while implementation lanes progress independently.

## State
- Lifecycle: `planned`
- Execution: `ready`
- Parallel policy: `parallel_safe`
- Result: `-`

## Next Action
Write tests that consume compiler-returned and native-constructed descriptors, compare semantics, and submit both through the same actual accelerator path after integration.

## Ownership
- `exclusive`: `tests/semantic_spine/verification/evidence_requirements.json`
- `exclusive`: `tests/semantic_spine/verification/integrated_probe.cc`
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
- `task`: `CE-SS1-V02`
<!-- todo-orchestrator:v2-managed:end -->
