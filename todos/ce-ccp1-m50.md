<!-- todo-orchestrator:v2-managed:start -->
# CE-CCP1-M50: Realization IR and CPU/NVIDIA backend foundation integrated

Task revision: `3895`; current project revision is in `todo-status.md`.

## Objective
Integrate and validate all P50 workstreams, freeze shared interfaces, and publish milestone M50.

## State
- Lifecycle: `planned`
- Execution: `ready`
- Parallel policy: `integration_exclusive`
- Result: `-`

## Next Action
Collect isolated lane receipts for P50, verify hashes and interfaces, integrate central files, run label ce_ccp1_m50, and publish CE-CCP1-MILESTONE-M50.

## Ownership
- `exclusive`: `cmake/compiler`
- `exclusive`: `cmake/providers`
- `exclusive`: `include/Cellerator/compiler/backend`
- `exclusive`: `include/Cellerator/compiler/ir/realization`
- `exclusive`: `src/compiler/backend`
- `forbidden`: `.todo-orchestrator`

## Dependencies
- `task`: `CE-CCP1-F01-018`
- `task`: `CE-CCP1-F02-014`
- `task`: `CE-CCP1-F03-015`
- `task`: `CE-CCP1-F04-013`
- `checkpoint`: `CE-CCP1-MILESTONE-M40`
<!-- todo-orchestrator:v2-managed:end -->
