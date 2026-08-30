

<!-- todo-orchestrator:v2-managed:start -->
# CE-GEO-107: Static contract audits

Task revision: `2999`; current project revision is in `todo-status.md`.

## Objective
Create a focused CE-GEO-only audit for new STL ownership, WMMA leakage, architecture in CSG1, CPK1 or CPE2 mutation, global fast math, atomics, and broad ownership; do not certify CE-PTR.

## State
- Lifecycle: `done`
- Execution: `closed`
- Parallel policy: `serial`
- Result: `validated`

## Next Action
_None._

## Ownership
- `exclusive`: `scripts/ce_geo/check_static_contracts.py`
- `exclusive`: `tests/ce_geo/validation/owned_production_paths.json`
- `exclusive`: `tests/ce_geo/validation/test_static_contracts.py`
- `read`: `include/Cellerator/compute/architecture`
- `read`: `include/Cellerator/compute/operation`
- `read`: `include/Cellerator/geometry`
- `read`: `include/Cellerator/runtime/device_descriptor.hh`
- `read`: `src/compute/architecture`
- `read`: `src/compute/projection`
- `read`: `src/geometry/compiler`
- `read`: `src/geometry/persistence`

## Dependencies
_None._
<!-- todo-orchestrator:v2-managed:end -->
