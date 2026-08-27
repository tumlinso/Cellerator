

<!-- todo-orchestrator:v2-managed:start -->
# CE-LIVE-32: One V100 Tensor Core candidate, promotion or measured rejection

Task revision: `2230`; current project revision is in `todo-status.md`.

## Objective
Implement at most one sm_70 dense-fragment or WMMA candidate under the bounded contract, integrate it as an ordinary planner candidate, and either promote on a complete-cost real-fixture win or leave unregistered with reproducible negative evidence.

## State
- Lifecycle: `done`
- Execution: `closed`
- Parallel policy: `parallel_safe`
- Result: `evaluated_not_promoted`

## Next Action
_None._

## Ownership
- `exclusive`: `bench/ce_live/tensor_core/campaign`
- `exclusive`: `docs/CE_LIVE_TENSOR_CORE.md`
- `exclusive`: `include/Cellerator/compute/math/operation_core/builtin_catalog.hh`
- `exclusive`: `include/Cellerator/compute/math/tensor_core`
- `exclusive`: `src/compute/math/operation_core/builtin_catalog.cc`
- `exclusive`: `src/compute/math/tensor_core`
- `exclusive`: `tests/tensor_core`
- `read`: `components/CellPack`
- `read`: `include/Cellerator/planner`

## Dependencies
- `task`: `CE-LIVE-16`
- `task`: `CE-LIVE-20`
- `task`: `CE-LIVE-21`
- `task`: `CE-LIVE-24`
- `task`: `CE-LIVE-30`
<!-- todo-orchestrator:v2-managed:end -->
