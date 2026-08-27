

<!-- todo-orchestrator:v2-managed:start -->
# CE-LIVE-40: CelleraTorch zero-copy tensor and parameter views

Task revision: `2165`; current project revision is in `todo-status.md`.

## Objective
Expose native Cellerator dense operands, value planes, and parameter descriptors to Torch with explicit lifetime ownership and correct device, shape, and stride metadata.

## State
- Lifecycle: `done`
- Execution: `closed`
- Parallel policy: `parallel_safe`
- Result: `implemented`

## Next Action
_None._

## Ownership
- `exclusive`: `components/CelleraTorch/docs/native_views.md`
- `exclusive`: `components/CelleraTorch/include/CelleraTorch/native_views.hh`
- `exclusive`: `components/CelleraTorch/src/native_views.cc`
- `exclusive`: `components/CelleraTorch/tests/native_views_test.cc`
- `read`: `include/Cellerator`

## Dependencies
- `task`: `CE-LIVE-37`
<!-- todo-orchestrator:v2-managed:end -->
