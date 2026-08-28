

<!-- todo-orchestrator:v2-managed:start -->
# CE-POST-REMAP-02: Localize CMake ownership

Task revision: `2360`; current project revision is in `todo-status.md`.

## Objective
Move implementation, test, benchmark, example, and tool target definitions to sensible subsystem-local CMake files while preserving target names, aliases, options, and runtime behavior.

## State
- Lifecycle: `done`
- Execution: `closed`
- Parallel policy: `serial`
- Result: `implemented`

## Next Action
Move target definitions conservatively by source ownership and keep subsystem fan-in explicit.

## Ownership
- `exclusive`: `CMakeLists.txt`
- `exclusive`: `bench/CMakeLists.txt`
- `exclusive`: `compat/CMakeLists.txt`
- `exclusive`: `components/CelleraTorch/CMakeLists.txt`
- `exclusive`: `examples/CMakeLists.txt`
- `exclusive`: `src`
- `exclusive`: `tests/CMakeLists.txt`
- `exclusive`: `tools/CMakeLists.txt`
- `read`: `bench`
- `read`: `components`
- `read`: `docs`
- `read`: `examples`
- `read`: `include`
- `read`: `tests`
- `read`: `tools`

## Dependencies
- `task`: `CE-POST-REMAP-01`
<!-- todo-orchestrator:v2-managed:end -->
