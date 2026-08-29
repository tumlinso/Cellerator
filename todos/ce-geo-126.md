

<!-- todo-orchestrator:v2-managed:start -->
# CE-GEO-126: Verify loaded CE-AMP extension and interlock

Task revision: `2379`; current project revision is in `todo-status.md`.

## Objective
Verify the complete CE-AMP run/lanes/tasks exist, every lane head requires CE-GEO-COMPLETE and human permission, no CE-AMP task started, and permission remains not_granted absent explicit user change.

## State
- Lifecycle: `planned`
- Execution: `ready`
- Parallel policy: `serial`
- Result: `-`

## Next Action
_None._

## Ownership
- `exclusive`: `bench/ce_geo/evidence/amp_interlock_audit.json`
- `read`: `.todo-orchestrator`
- `read`: `ce-geo-plan.json`

## Dependencies
- `task`: `CE-GEO-125`
<!-- todo-orchestrator:v2-managed:end -->
