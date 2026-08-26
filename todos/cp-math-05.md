<!-- todo-orchestrator:v2-managed:start -->
# CP-MATH-05: Native warp-tile math adapter and sidecars

Task revision: `1418`; current project revision is in `todo-status.md`.

## Objective
Expose the existing frozen plan/order/warp tiles/CPK1 to math with derived union masks, packed offsets, density/reuse/workload sidecars and an exact decoder; copy no compact values and invent no row permutation.

## State
- Lifecycle: `superseded`
- Execution: `closed`
- Parallel policy: `serial`
- Result: `superseded`

## Next Action
_None._

## Ownership
- `exclusive`: `Cellerator/include/Cellerator/compute/math/native_tile_view.hh`
- `exclusive`: `Cellerator/src/compute/math/native_tile_view.cc`
- `exclusive`: `Cellerator/tests/math_native_tile_adapter_test.cc`
- `forbidden`: `Cellerator/components/CellPack`
- `read`: `Cellerator/components/CellPack/include/CellPack/local_cell_ordering.hh`
- `read`: `Cellerator/components/CellPack/include/CellPack/packing_plan.hh`
- `read`: `Cellerator/components/CellPack/include/CellPack/persistent_packing_payload.hh`
- `read`: `Cellerator/components/CellPack/include/CellPack/warp_tiles.hh`

## Dependencies
_None._
<!-- todo-orchestrator:v2-managed:end -->
