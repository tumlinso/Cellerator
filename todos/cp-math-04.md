

<!-- todo-orchestrator:v2-managed:start -->
# CP-MATH-04: cuSPARSE Blocked-ELL lowering

Task revision: `1418`; current project revision is in `todo-status.md`.

## Objective
Lower unchanged variable-width semantic blocks and local row order into legal BELL8/16/32 candidates, record occupancy/utilization/expansion/storage, reject absurd candidates, and validate via independent decode.

## State
- Lifecycle: `superseded`
- Execution: `closed`
- Parallel policy: `serial`
- Result: `superseded`

## Next Action
_None._

## Ownership
- `exclusive`: `Cellerator/include/Cellerator/compute/math/physical_bell.hh`
- `exclusive`: `Cellerator/src/compute/math/physical_bell.cc`
- `exclusive`: `Cellerator/src/compute/math/physical_bell_candidates.cc`
- `exclusive`: `Cellerator/tests/math_bell_lowering_test.cc`
- `forbidden`: `Cellerator/components/CellPack`
- `read`: `Cellerator/components/CellPack/include/CellPack/local_cell_ordering.hh`
- `read`: `Cellerator/components/CellPack/include/CellPack/packing_plan.hh`

## Dependencies
_None._
<!-- todo-orchestrator:v2-managed:end -->
