<!-- todo-orchestrator:v2-managed:start -->
# CE-LIVE-16: Bounded Tensor Core feasibility and candidate contract

Task revision: `1423`; current project revision is in `todo-status.md`.

## Objective
Define one V100-relevant dense-fragment or WMMA candidate lane, including density classification, packing, alignment, tails, numeric policy, forward/backward maps, complete planner costs, and rejection criteria, without modifying common semantic ABIs or registering a kernel.

## State
- Lifecycle: `planned`
- Execution: `ready`
- Parallel policy: `parallel_safe`
- Result: `-`

## Next Action
_None._

## Ownership
- `exclusive`: `bench/ce_live/tensor_core/contract`
- `exclusive`: `docs/CE_LIVE_TENSOR_CORE.md`
- `read`: `bench/architecture_evidence`
- `read`: `components/CellPack`
- `read`: `include/Cellerator/compute/math`
- `read`: `include/Cellerator/planner`

## Dependencies
- `task`: `CE-LIVE-01`
<!-- todo-orchestrator:v2-managed:end -->
