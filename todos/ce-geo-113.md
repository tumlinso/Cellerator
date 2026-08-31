

<!-- todo-orchestrator:v2-managed:start -->
# CE-GEO-113: First complete-cost hybrid forward campaign

Task revision: `3281`; current project revision is in `todo-status.md`.

## Objective
Measure the integrated N64 path including semantic search, refinement, projection/upload, preparation, value/input pack, MMA, residual, epilogue, order, synchronization, reuse, and best sparse fallback; produce CE-GEO-79 evidence.

## State
- Lifecycle: `done`
- Execution: `closed`
- Parallel policy: `serial`
- Result: `validated`

## Next Action
_None._

## Ownership
- `exclusive`: `bench/ce_geo/evidence/sm70_forward_complete_cost.jsonl`
- `exclusive`: `bench/tensor_core/ce_geo/hybrid_forward.cu`
- `exclusive`: `bench/tensor_core/ce_geo/run_hybrid_forward.py`
- `read`: `bench/ce_geo/harness`
- `read`: `src/compute/architecture/providers/nvidia/sm70`
- `read`: `src/compute/projection`
- `read`: `src/planner`

## Dependencies
- `checkpoint`: `CE-GEO-SM70-N64-VERTICAL`
<!-- todo-orchestrator:v2-managed:end -->
