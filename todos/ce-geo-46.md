

<!-- todo-orchestrator:v2-managed:start -->
# CE-GEO-46: Acquisition diagnostics and planner lifetime mapping

Task revision: `3016`; current project revision is in `todo-status.md`.

## Objective
Map semantic search, refinement, construction, upload, prebind, preparation, value pack, input pack, kernel, epilogue, and order work into existing planner-v2 phases with reuse diagnostics.

## State
- Lifecycle: `done`
- Execution: `closed`
- Parallel policy: `serial`
- Result: `implemented`

## Next Action
_None._

## Ownership
- `exclusive`: `include/Cellerator/execution/geometry_acquisition_diagnostics.hh`
- `exclusive`: `src/execution/geometry_acquisition_diagnostics.cc`
- `exclusive`: `tests/ce_geo/persistence/acquisition_cost_mapping_test.cc`
- `read`: `include/Cellerator/planner/candidate_measurement.hh`
- `read`: `src/planner/candidate_measurement.cu`

## Dependencies
_None._
<!-- todo-orchestrator:v2-managed:end -->
