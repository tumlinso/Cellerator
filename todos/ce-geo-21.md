

<!-- todo-orchestrator:v2-managed:start -->
# CE-GEO-21: Core five-candidate compatibility fragment

Task revision: `3281`; current project revision is in `todo-status.md`.

## Objective
Lift the current five built-in candidates into a catalog-v2 compatibility fragment without changing behavior or identities.

## State
- Lifecycle: `done`
- Execution: `closed`
- Parallel policy: `serial`
- Result: `implemented`

## Next Action
_None._

## Ownership
- `exclusive`: `include/Cellerator/compute/operation/builtin_catalog.hh`
- `exclusive`: `src/compute/operation/builtin_catalog.cc`
- `exclusive`: `tests/ce_geo/catalog/builtin_fragment_test.cu`
- `read`: `include/Cellerator/compute/candidate`
- `read`: `src/compute/candidate`

## Dependencies
_None._
<!-- todo-orchestrator:v2-managed:end -->
