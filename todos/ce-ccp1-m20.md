

<!-- todo-orchestrator:v2-managed:start -->
# CE-CCP1-M20: Source language parser, AST, Sema, and execution-field semantics integrated

Task revision: `6550`; current project revision is in `todo-status.md`.

## Objective
Integrate and validate all P20 workstreams, freeze shared interfaces, and publish milestone M20.

## State
- Lifecycle: `done`
- Execution: `closed`
- Parallel policy: `integration_exclusive`
- Result: `validated`

## Next Action
Collect isolated lane receipts for P20, verify hashes and interfaces, integrate central files, run label ce_ccp1_m20, and publish CE-CCP1-MILESTONE-M20.

## Ownership
- `exclusive`: `include/Cellerator/compiler.hh`
- `exclusive`: `include/Cellerator/compiler/ast`
- `exclusive`: `include/Cellerator/compiler/frontend`
- `exclusive`: `include/Cellerator/compiler/sema`
- `exclusive`: `src/compiler/CMakeLists.txt`
- `forbidden`: `.todo-orchestrator`

## Dependencies
- `task`: `CE-CCP1-C01-016`
- `task`: `CE-CCP1-C02-012`
- `task`: `CE-CCP1-C03-016`
- `task`: `CE-CCP1-C04-016`
- `checkpoint`: `CE-CCP1-MILESTONE-M10`
<!-- todo-orchestrator:v2-managed:end -->
