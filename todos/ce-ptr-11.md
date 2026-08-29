

<!-- todo-orchestrator:v2-managed:start -->
# CE-PTR-11: Exact-search internal fixed-K representation

Task revision: `2376`; current project revision is in `todo-status.md`.

## Objective
Preserve raw public device views while evaluating and implementing K-specialized internal top-K storage and merges with controlled register pressure, spills, shared-memory use, warp cooperation, and compact identities.

## State
- Lifecycle: `planned`
- Execution: `ready`
- Parallel policy: `parallel_safe`
- Result: `-`

## Next Action
Measure current V100 register and spill behavior, compare bounded implementation alternatives, and promote only evidence-backed internal forms.

## Ownership
- `exclusive`: `src/compute/neighbors/exact_search`
- `read`: `bench`
- `read`: `include/Cellerator/compute/neighbors`
- `read`: `tests`

## Dependencies
- `checkpoint`: `CE-PTR-SUBSTRATE-CONTRACT-READY`
<!-- todo-orchestrator:v2-managed:end -->
