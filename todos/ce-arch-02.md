

<!-- todo-orchestrator:v2-managed:start -->
# CE-ARCH-02: Immediate correctness containment and experimental quarantine

Task revision: `780`; current project revision is in `todo-status.md`.

## Objective
Contain unsafe experimental paths before architectural expansion, without performance redesign or new kernels.

## State
- Lifecycle: `done`
- Execution: `closed`
- Parallel policy: `serial`
- Result: `validated`

## Next Action
After explicit implementation authorization, reject unsupported padded BELL and stale bindings and label non-integrated CP-Math paths experimental; do not design a new kernel or ABI.

## Ownership
- `exclusive`: `Cellerator/include/Cellerator/compute/math`
- `exclusive`: `Cellerator/src/compute/math`
- `exclusive`: `Cellerator/tests/math_bell_lowering_test.cc`
- `exclusive`: `Cellerator/tests/math_cusparse_bell_test.cu`
- `exclusive`: `Cellerator/tests/math_operation_contract_test.cc`
- `forbidden`: `CellShard`
- `read`: `Baseplane/include/Baseplane/seq`
- `read`: `Baseplane/tests/seq`
- `read`: `Cellerator/include/Cellerator/runtime`
- `read`: `Cellerator/src/runtime`

## Dependencies
- `task`: `CE-ARCH-01`
<!-- todo-orchestrator:v2-managed:end -->
