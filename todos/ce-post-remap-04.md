

<!-- todo-orchestrator:v2-managed:start -->
# CE-POST-REMAP-04: Clean validation and workflow reconciliation

Task revision: `2363`; current project revision is in `todo-status.md`.

## Objective
Run fresh native, CelleraTorch, independent CellShard, integration, CE-LIVE, sanitizer, compatibility, layout, suffix, include, and dependency-graph validation; record exact evidence; reconcile the historical CE-REMAP run through authority; and leave both repositories clean.

## State
- Lifecycle: `in_progress`
- Execution: `claimed`
- Parallel policy: `integration_exclusive`
- Result: `-`

## Next Action
Validate from fresh build directories, record exact graph and test evidence, then reconcile the historical CE-REMAP workflow state.

## Ownership
- `exclusive`: `.todo-orchestrator`
- `exclusive`: `bench/repository_remap/post_remap`
- `exclusive`: `cellerator-post-remap-plan.json`
- `exclusive`: `docs/REPOSITORY_CONSOLIDATION_PROGRAM.md`
- `exclusive`: `docs/current_implementation.qmd`
- `exclusive`: `docs/migration_roadmap.qmd`
- `exclusive`: `todo-status.md`
- `exclusive`: `todos.md`
- `read`: `AGENTS.md`
- `read`: `CMakeLists.txt`
- `read`: `bench`
- `read`: `cmake`
- `read`: `compat`
- `read`: `components`
- `read`: `docs`
- `read`: `examples`
- `read`: `include`
- `read`: `modules`
- `read`: `scripts`
- `read`: `src`
- `read`: `tests`
- `read`: `tools`

## Dependencies
- `task`: `CE-POST-REMAP-02`
- `task`: `CE-POST-REMAP-03`
<!-- todo-orchestrator:v2-managed:end -->
