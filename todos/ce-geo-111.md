

<!-- todo-orchestrator:v2-managed:start -->
# CE-GEO-111: Value-pack input-order residual and remap calibration

Task revision: `3016`; current project revision is in `todo-status.md`.

## Objective
Calibrate standalone value pack, dense input layout, residual, epilogue, and output remap across sizes and reuse with exact complete-phase evidence.

## State
- Lifecycle: `done`
- Execution: `closed`
- Parallel policy: `parallel_safe`
- Result: `validated`

## Next Action
_None._

## Ownership
- `exclusive`: `bench/ce_geo/evidence/micro/value_pack_residual.jsonl`
- `exclusive`: `bench/tensor_core/ce_geo/value_pack_residual_calibration.cu`
- `read`: `bench/ce_geo/harness`
- `read`: `src/compute/architecture/providers/nvidia/sm70/residual.cu`
- `read`: `src/compute/architecture/providers/nvidia/sm70/value_pack.cu`

## Dependencies
- `checkpoint`: `CE-GEO-BENCH-INFRA-V1`
- `task`: `CE-GEO-72`
<!-- todo-orchestrator:v2-managed:end -->
