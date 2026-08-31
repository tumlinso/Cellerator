

<!-- todo-orchestrator:v2-managed:start -->
# CE-GEO-96: Sparse biological exchange composition

Task revision: `3281`; current project revision is in `todo-status.md`.

## Objective
Compose support contraction, edge mapping/gating, segment normalization, and relation apply as separate operations under one prepared execution context.

## State
- Lifecycle: `done`
- Execution: `closed`
- Parallel policy: `serial`
- Result: `implemented`

## Next Action
_None._

## Ownership
- `exclusive`: `src/compute/architecture/providers/nvidia/sm70/exchange_program.cc`
- `exclusive`: `tests/tensor_core/sm70/exchange_program_test.cu`
- `read`: `include/Cellerator/compute/operation/relation_algebra.hh`
- `read`: `src/compute/architecture/providers/nvidia/sm70/contract_on_support.cu`

## Dependencies
_None._
<!-- todo-orchestrator:v2-managed:end -->
