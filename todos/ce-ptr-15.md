

<!-- todo-orchestrator:v2-managed:start -->
# CE-PTR-15: Repository-wide final migration and acceptance

Task revision: `2522`; current project revision is in `todo-status.md`.

## Objective
Converge CE-PTR by removing obsolete generic infrastructure and stale owners, resolving the production inventory to documented exceptions, running comprehensive semantic, allocation, synchronization, transfer, performance, compiler, persistence, sanitizer, and source-tooling acceptance, and updating durable architecture documentation.

## State
- Lifecycle: `done`
- Execution: `closed`
- Parallel policy: `project_exclusive`
- Result: `implemented`

## Next Action
After all prior workstreams and CE-PTR-13 are complete, run the project-exclusive convergence audit and close CE-PTR only on complete evidence.

## Ownership
- `exclusive`: `bench/ce_ptr/final`
- `exclusive`: `docs/CE_PTR_INVENTORY.md`
- `exclusive`: `docs/CE_PTR_POLICY.md`
- `exclusive`: `docs/architecture.qmd`
- `exclusive`: `docs/current_implementation.qmd`
- `exclusive`: `docs/migration_roadmap.qmd`
- `exclusive`: `docs/performance_validation.qmd`
- `exclusive`: `scripts/check_no_inappropriate_core_stl.py`
- `read`: `.todo-orchestrator`
- `read`: `AGENTS.md`
- `read`: `CMakeLists.txt`
- `read`: `bench`
- `read`: `compat`
- `read`: `components`
- `read`: `docs`
- `read`: `include`
- `read`: `scripts`
- `read`: `src`
- `read`: `tests`

## Dependencies
- `barrier`: `CE-PTR-FINAL-FANIN`
<!-- todo-orchestrator:v2-managed:end -->
