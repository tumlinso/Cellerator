

<!-- todo-orchestrator:v2-managed:start -->
# CE-GEO-122: Integrate biology operation substrate

Task revision: `3281`; current project revision is in `todo-status.md`.

## Objective
Integrate relation algebra, transpose, contraction, segments, gradients, bundles, examples, exchange, and explicitly disposed optional fusion.

## State
- Lifecycle: `done`
- Execution: `closed`
- Parallel policy: `integration_exclusive`
- Result: `implemented`

## Next Action
_None._

## Ownership
- `exclusive`: `examples/CMakeLists.txt`
- `exclusive`: `examples/ce_geo/CMakeLists.txt`
- `exclusive`: `src/compute/CMakeLists.txt`
- `exclusive`: `src/compute/candidate/segment/CMakeLists.txt`
- `exclusive`: `src/compute/operation/relation_algebra_assembly.cc`
- `exclusive`: `tests/CMakeLists.txt`
- `exclusive`: `tests/ce_geo/run_biology_suite.py`
- `exclusive`: `tests/relation_algebra/CMakeLists.txt`
- `read`: `examples/ce_geo`
- `read`: `include/Cellerator/compute/operation`
- `read`: `src/compute/architecture/providers/nvidia/sm70`
- `read`: `src/compute/candidate/segment`

## Dependencies
- `barrier`: `CE-GEO-BIOLOGY-FANIN`
<!-- todo-orchestrator:v2-managed:end -->
