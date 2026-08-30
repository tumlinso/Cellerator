

<!-- todo-orchestrator:v2-managed:start -->
# CE-PTR-13: Runtime buffer, scratch, and ownership cleanup

Task revision: `2520`; current project revision is in `todo-status.md`.

## Objective
After consumers migrate, remove obsolete host_buffer, shared-owned device buffers, duplicate graph-local buffers, grow-by-reallocation scratch arenas, and blocking copy helpers, converging on session allocation handles, explicit views and streams, prepared scratch, and stable addresses.

## State
- Lifecycle: `done`
- Execution: `closed`
- Parallel policy: `integration_exclusive`
- Result: `implemented`

## Next Action
Reinventory all consumers after the implementation fan-in, remove only truly obsolete infrastructure, and run integration and graph-capture validation.

## Ownership
- `exclusive`: `docs/compute_and_models.qmd`
- `exclusive`: `include/Cellerator/compute/core/host_buffer.hh`
- `exclusive`: `include/Cellerator/runtime`
- `exclusive`: `include/Cellerator/runtime/device_buffer.cuh`
- `exclusive`: `src/compute/current_targets.cmake`
- `exclusive`: `src/geometry/optimizer.cc`
- `exclusive`: `src/geometry/optimizer_state.hh`
- `exclusive`: `src/runtime`
- `exclusive`: `tests/CMakeLists.txt`
- `read`: `bench`
- `read`: `compat`
- `read`: `components/CelleraTorch`
- `read`: `include/Cellerator`
- `read`: `src`
- `read`: `tests`

## Dependencies
- `barrier`: `CE-PTR-IMPLEMENTATION-FANIN`
<!-- todo-orchestrator:v2-managed:end -->
