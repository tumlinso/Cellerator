

<!-- todo-orchestrator:v2-managed:start -->
# CE-GEO-114: Width occupancy reuse and resource sweeps

Task revision: `2379`; current project revision is in `todo-status.md`.

## Objective
Sweep N 1,4,8,16,32,64,128,256,512; D 16,32,64,128,256,512; reuse 1,4,16,64,256,1000+; record registers, shared memory, occupancy, bandwidth, caches, stalls, launches, useful/executed work, residuals, and errors.

## State
- Lifecycle: `planned`
- Execution: `ready`
- Parallel policy: `serial`
- Result: `-`

## Next Action
_None._

## Ownership
- `exclusive`: `bench/ce_geo/evidence/micro/width_reuse_sweep.jsonl`
- `exclusive`: `bench/tensor_core/ce_geo/width_reuse_sweep.cu`
- `read`: `bench/ce_geo/harness`
- `read`: `src/compute/architecture/providers/nvidia/sm70`

## Dependencies
_None._
<!-- todo-orchestrator:v2-managed:end -->
