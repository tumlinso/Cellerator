

<!-- todo-orchestrator:v2-managed:start -->
# CE-LIVE-11: Freeze logical relation orientation and edge identity

Task revision: `2185`; current project revision is in `todo-status.md`.

## Objective
Resolve the feature/row orientation seam so forward relations are feature-or-gene source to row-or-cell destination, transpose projections share the same logical edge identity, and swapped axes fail explicitly.

## State
- Lifecycle: `done`
- Execution: `closed`
- Parallel policy: `parallel_safe`
- Result: `implemented`

## Next Action
_None._

## Ownership
- `exclusive`: `components/CellPack/include/CellPack/semantic_geometry.hh`
- `exclusive`: `components/CellPack/src/semantic_geometry.cc`
- `exclusive`: `components/CellPack/tests/semantic_geometry_adapter_test.cc`
- `exclusive`: `docs/CE_LIVE_RELATION_ORIENTATION.md`
- `exclusive`: `include/Cellerator/compute/math/native_tile_view.hh`
- `exclusive`: `src/compute/math/native_tile_view.cc`
- `exclusive`: `tests/math_core/feature_major_small_n_candidate_test.cu`
- `exclusive`: `tests/math_core/row_masked_n1_candidate_test.cu`
- `exclusive`: `tests/math_core/transpose_backward_candidate_test.cu`
- `exclusive`: `tests/math_native_tile_adapter_test.cc`
- `read`: `components/CellPack`
- `read`: `include/Cellerator/execution`
- `read`: `src/compute/math/operation_core`

## Dependencies
- `task`: `CE-LIVE-01`
<!-- todo-orchestrator:v2-managed:end -->
