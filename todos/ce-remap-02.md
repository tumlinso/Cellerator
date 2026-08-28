

<!-- todo-orchestrator:v2-managed:start -->
# CE-REMAP-02: Canonical skeleton, naming checks, and bounded module probe

Task revision: `2366`; current project revision is in `todo-status.md`.

## Objective
Create canonical subsystem build ownership and repository layout checks, prove direct Clang 18 module import, and record that native CMake scanning is deferred because the required generator and scanner are unavailable.

## State
- Lifecycle: `done`
- Execution: `closed`
- Parallel policy: `integration_exclusive`
- Result: `implemented`

## Next Action
_None._

## Ownership
- `exclusive`: `CMakeLists.txt`
- `exclusive`: `cmake`
- `exclusive`: `compat`
- `exclusive`: `docs/REPOSITORY_CONSOLIDATION_PROGRAM.md`
- `exclusive`: `examples`
- `exclusive`: `modules`
- `exclusive`: `scripts/check_repository_layout.py`
- `exclusive`: `tests/modules`
- `exclusive`: `tools`
- `read`: `bench`
- `read`: `components`
- `read`: `docs`
- `read`: `include`
- `read`: `python`
- `read`: `src`
- `read`: `tests`

## Dependencies
- `task`: `CE-REMAP-01`
<!-- todo-orchestrator:v2-managed:end -->
