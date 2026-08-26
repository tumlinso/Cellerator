<!-- todo-orchestrator:v2-managed:start -->
# CE-ARCH-40: Cellerator-owned Baseplane ABI and sequence-state integration

Task revision: `780`; current project revision is in `todo-status.md`.

## Objective
Make Baseplane sequence structures native Cellerator operands without a host, dense-matrix, or generic-SpMM boundary.

## State
- Lifecycle: `done`
- Execution: `closed`
- Parallel policy: `serial`
- Result: `implemented`

## Next Action
Complete. The frozen v1 seam imports Baseplane predicate/event contracts, exposes validity-aware sequence operands, and validates fused and caller-materialized predicate-to-regulatory-to-gene execution on V100 sm_70.

## Ownership
- `exclusive`: `Cellerator/CMakeLists.txt`
- `exclusive`: `Cellerator/include/Cellerator/compute/sequence`
- `exclusive`: `Cellerator/src/compute/sequence`
- `exclusive`: `Cellerator/tests/sequence`
- `forbidden`: `CellShard`
- `forbidden`: `Cellerator/components/CelleraTorch`
- `read`: `Baseplane/include/Baseplane/seq`
- `read`: `Baseplane/src/seq`
- `read`: `Baseplane/tests/seq`

## Dependencies
- `task`: `CE-ARCH-10`
- `task`: `CE-ARCH-11`
- `task`: `CE-ARCH-12`
- `task`: `CE-ARCH-22`
- `decision`: `BASEPLANE-CORRECTNESS-EXTERNAL`
- `decision`: `BASEPLANE-PREPARED-PLAN-EXTERNAL`
<!-- todo-orchestrator:v2-managed:end -->
