

<!-- todo-orchestrator:v2-managed:start -->
# CE-GEO-106: Stream graph generation and hot-path validation

Task revision: `2999`; current project revision is in `todo-status.md`.

## Objective
Prove post-seal no allocation, query, image parse, provider search, descriptor build, hidden sync, or canonicalization; validate external streams, graph addresses, repeated generations, and concurrent plans.

## State
- Lifecycle: `done`
- Execution: `closed`
- Parallel policy: `serial`
- Result: `validated`

## Next Action
_None._

## Ownership
- `exclusive`: `tests/ce_geo/validation/hot_path_test.cu`
- `read`: `src/compute/architecture/providers/nvidia/sm70`
- `read`: `src/execution`
- `read`: `src/runtime`

## Dependencies
_None._
<!-- todo-orchestrator:v2-managed:end -->
