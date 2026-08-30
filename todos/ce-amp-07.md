

<!-- todo-orchestrator:v2-managed:start -->
# CE-AMP-07: Ampere calibration and local baselines

Task revision: `3016`; current project revision is in `todo-status.md`.

## Objective
Benchmark complete costs against the strongest A5000-local sparse, dense, and vendor baselines with exact hardware/toolchain/profiler evidence.

## State
- Lifecycle: `planned`
- Execution: `ready`
- Parallel policy: `parallel_safe`
- Result: `-`

## Next Action
_None._

## Ownership
- `exclusive`: `bench/ce_geo/evidence/ampere/calibration.jsonl`
- `exclusive`: `bench/tensor_core/ce_geo/sm86_calibration.cu`
- `read`: `bench/ce_geo/harness`
- `read`: `src/compute/architecture/providers/nvidia/sm86`

## Dependencies
- `checkpoint`: `CE-GEO-COMPLETE`
- `decision`: `CE-AMP-PERMISSION`
- `task`: `CE-AMP-04`
- `checkpoint`: `CE-EXOP-COMPLETE`
<!-- todo-orchestrator:v2-managed:end -->
