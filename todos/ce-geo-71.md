

<!-- todo-orchestrator:v2-managed:start -->
# CE-GEO-71: sm_70 provider and capability shell

Task revision: `2379`; current project revision is in `todo-status.md`.

## Objective
Add the source-linked nvidia_sm70 provider advertising only implemented FP16 relation/input, FP32 accumulate/output 16x16x16 WMMA.

## State
- Lifecycle: `planned`
- Execution: `ready`
- Parallel policy: `serial`
- Result: `-`

## Next Action
_None._

## Ownership
- `exclusive`: `include/Cellerator/compute/architecture/providers/nvidia/sm70_provider.hh`
- `exclusive`: `src/compute/architecture/providers/nvidia/sm70/provider.cc`
- `exclusive`: `tests/tensor_core/sm70/provider_test.cu`
- `read`: `include/Cellerator/compute/architecture/provider.hh`
- `read`: `include/Cellerator/runtime/device_descriptor.hh`

## Dependencies
_None._
<!-- todo-orchestrator:v2-managed:end -->
