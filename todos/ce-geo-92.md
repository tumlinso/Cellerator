

<!-- todo-orchestrator:v2-managed:start -->
# CE-GEO-92: Support-contraction projection adapter

Task revision: `3281`; current project revision is in `todo-status.md`.

## Objective
Prepare hybrid projection views for support-restricted destination/source contractions with stable edge output.

## State
- Lifecycle: `done`
- Execution: `closed`
- Parallel policy: `serial`
- Result: `implemented`

## Next Action
_None._

## Ownership
- `exclusive`: `src/compute/architecture/providers/nvidia/sm70/contract_on_support_projection.cc`
- `exclusive`: `tests/tensor_core/sm70/contract_projection_test.cc`
- `read`: `include/Cellerator/compute/projection/physical_mma_hybrid.hh`

## Dependencies
_None._
<!-- todo-orchestrator:v2-managed:end -->
