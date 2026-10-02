

<!-- todo-orchestrator:v2-managed:start -->
# CE-MOON-010: Adopt the numerical research lane without replacing ML2

Task revision: `7478`; current project revision is in `todo-status.md`.

## Objective
Inspect current Cellerator claims and numerical entry points; install a small independently buildable numerical experimental module and split supplied mixed research seeds by semantic ownership. Preserve all existing ML2 and unrelated work. No production runtime integration is a prerequisite.

## State
- Lifecycle: `done`
- Execution: `closed`
- Parallel policy: `integration_exclusive`
- Result: `implemented`

## Next Action
Read planning/baseplane_moonshot_bootstrap/tasks/CE-MOON-010.md; implement concrete research operators under Cellerator authority. Apply the adopted ownership and research policy recorded in planning/baseplane_moonshot_adoption/cellerator-policy.todo-plan.json; verify its authority receipt before seed installation.

## Ownership
- `exclusive`: `experiments/baseplane_moonshot/CMakeLists.txt`
- `exclusive`: `experiments/baseplane_moonshot/README.md`
- `exclusive`: `experiments/baseplane_moonshot/cuda`
- `exclusive`: `experiments/baseplane_moonshot/host`
- `exclusive`: `experiments/baseplane_moonshot/include`
- `exclusive`: `experiments/baseplane_moonshot/python`
- `exclusive`: `planning/baseplane_moonshot_bootstrap/results/cellerator`
- `forbidden`: `.todo-orchestrator`
- `read`: `docs/learning`
- `read`: `examples/relation_update_spine_v1`
- `read`: `experiments/baseplane_moonshot`
- `read`: `include/Cellerator`
- `read`: `planning/baseplane_moonshot_bootstrap`
- `read`: `src/compiler`
- `read`: `src/execution`
- `read`: `src/geometry`
- `read`: `tests/native_foundation/training`

## Dependencies
_None._
<!-- todo-orchestrator:v2-managed:end -->
