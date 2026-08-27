

<!-- todo-orchestrator:v2-managed:start -->
# CE-LIVE-42: CelleraTorch autograd and readiness adapter

Task revision: `2087`; current project revision is in `todo-status.md`.

## Objective
Connect Torch autograd to the native training executable, propagate readiness and current-stream dependencies correctly, expose native learned parameters without copying ownership into Torch, and test forward/backward/update parity.

## State
- Lifecycle: `planned`
- Execution: `ready`
- Parallel policy: `parallel_safe`
- Result: `-`

## Next Action
_None._

## Ownership
- `exclusive`: `components/CelleraTorch/docs/autograd_ops.md`
- `exclusive`: `components/CelleraTorch/include/CelleraTorch/autograd_ops.hh`
- `exclusive`: `components/CelleraTorch/src/autograd_ops.cu`
- `exclusive`: `components/CelleraTorch/tests/autograd_ops_test.cc`
- `read`: `include/Cellerator/execution/training_program.hh`
- `read`: `include/Cellerator/runtime/value_readiness.cuh`

## Dependencies
- `task`: `CE-LIVE-37`
<!-- todo-orchestrator:v2-managed:end -->
