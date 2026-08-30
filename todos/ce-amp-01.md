

<!-- todo-orchestrator:v2-managed:start -->
# CE-AMP-01: sm_86 device and provider capability records

Task revision: `2999`; current project revision is in `todo-status.md`.

## Objective
Advertise only source-linked m16n8k16 FP16xFP16 to FP32 and BF16xBF16 to FP32 capabilities with separate memory interfaces.

## State
- Lifecycle: `planned`
- Execution: `ready`
- Parallel policy: `parallel_safe`
- Result: `-`

## Next Action
_None._

## Ownership
- `exclusive`: `include/Cellerator/compute/architecture/providers/nvidia/sm86_provider.hh`
- `exclusive`: `src/compute/architecture/providers/nvidia/sm86/provider.cc`
- `exclusive`: `tests/tensor_core/sm86/provider_test.cu`
- `read`: `include/Cellerator/compute/architecture/provider.hh`
- `read`: `include/Cellerator/compute/architecture/providers/nvidia/sm70_provider.hh`

## Dependencies
- `checkpoint`: `CE-GEO-COMPLETE`
- `decision`: `CE-AMP-PERMISSION`
<!-- todo-orchestrator:v2-managed:end -->
