

<!-- todo-orchestrator:v2-managed:start -->
# CE-LIVE-14: Value readiness and asynchronous generation contract

Task revision: `2230`; current project revision is in `todo-status.md`.

## Objective
Design and implement the runtime-side generation-readiness token or record with same-stream fast paths, cross-stream waits, failed-enqueue safety, and no event or stream ownership in the persistent biological ABI.

## State
- Lifecycle: `done`
- Execution: `closed`
- Parallel policy: `parallel_safe`
- Result: `implemented`

## Next Action
_None._

## Ownership
- `exclusive`: `docs/CE_LIVE_VALUE_READINESS.md`
- `exclusive`: `include/Cellerator/runtime/value_readiness.cuh`
- `exclusive`: `src/runtime/value_readiness.cu`
- `exclusive`: `tests/runtime/value_readiness_test.cu`
- `read`: `include/Cellerator/execution/launch_bindings.hh`
- `read`: `include/Cellerator/execution/lifetimes.hh`
- `read`: `include/Cellerator/runtime/session.cuh`

## Dependencies
- `task`: `CE-LIVE-12`
<!-- todo-orchestrator:v2-managed:end -->
