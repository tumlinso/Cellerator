

<!-- todo-orchestrator:v2-managed:start -->
# CE-REMAP-04: Consolidate CellPack and CP-BP into geometry

Task revision: `2292`; current project revision is in `todo-status.md`.

## Objective
Move validated CellPack and related packing/discovery implementation into canonical geometry ownership, preserve namespaces and behavior, and leave compatibility aliases only where real consumers require them.

## State
- Lifecycle: `done`
- Execution: `closed`
- Parallel policy: `integration_exclusive`
- Result: `implemented`

## Next Action
_None._

## Ownership
- `exclusive`: `CMakeLists.txt`
- `exclusive`: `bench/CMakeLists.txt`
- `exclusive`: `bench/geometry`
- `exclusive`: `components/CellPack`
- `exclusive`: `docs/REPOSITORY_CONSOLIDATION_PROGRAM.md`
- `exclusive`: `include/CellPack`
- `exclusive`: `include/Cellerator/compute/gene_candidate_discovery.hh`
- `exclusive`: `include/Cellerator/compute/gene_support_bitset.hh`
- `exclusive`: `include/Cellerator/geometry`
- `exclusive`: `src/compute/packing`
- `exclusive`: `src/geometry`
- `exclusive`: `src/geometry/CMakeLists.txt`
- `exclusive`: `tests/CMakeLists.txt`
- `exclusive`: `tests/geometry`
- `read`: `bench`
- `read`: `components/CelleraTorch`
- `read`: `docs`
- `read`: `include`
- `read`: `src`
- `read`: `tests`

## Dependencies
- `task`: `CE-REMAP-03`
<!-- todo-orchestrator:v2-managed:end -->
