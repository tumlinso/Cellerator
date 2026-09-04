

<!-- todo-orchestrator:v2-managed:start -->
# CE-CCP1-M70: Cross-TU/LTO, libCellerator, standard library, and installable SDK integrated

Task revision: `4123`; current project revision is in `todo-status.md`.

## Objective
Integrate and validate all P70 workstreams, freeze shared interfaces, and publish milestone M70.

## State
- Lifecycle: `planned`
- Execution: `ready`
- Parallel policy: `integration_exclusive`
- Result: `-`

## Next Action
Collect isolated lane receipts for P70, verify hashes and interfaces, integrate central files, run label ce_ccp1_m70, and publish CE-CCP1-MILESTONE-M70.

## Ownership
- `exclusive`: `CMakeLists.txt`
- `exclusive`: `cmake/package`
- `exclusive`: `include/Cellerator`
- `exclusive`: `profiles/reference`
- `exclusive`: `stdlib`
- `exclusive`: `tools/CMakeLists.txt`
- `forbidden`: `.todo-orchestrator`

## Dependencies
- `task`: `CE-CCP1-H01-016`
- `task`: `CE-CCP1-H02-016`
- `task`: `CE-CCP1-H03-018`
- `checkpoint`: `CE-CCP1-MILESTONE-M60`
<!-- todo-orchestrator:v2-managed:end -->
