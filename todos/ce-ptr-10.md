

<!-- todo-orchestrator:v2-managed:start -->
# CE-PTR-10: Trajectory and graph structures

Task revision: `2376`; current project revision is in `todo-status.md`.

## Objective
Replace generic trajectory graph and table ownership with SoA images, bounded fixed-width candidate or forward edges where natural, tree topology and child CSR projections, Euler order and inverse, two-pass supernodes, packed aggregation, and allocation-aware branch detection.

## State
- Lifecycle: `planned`
- Execution: `ready`
- Parallel policy: `parallel_safe`
- Result: `-`

## Next Action
Classify each graph by actual bounds and traversal before selecting fixed-width, CSR, Euler, or packed-relation forms.

## Ownership
- `exclusive`: `include/Cellerator/compute/operators/graph`
- `exclusive`: `include/Cellerator/examples/trajectory`
- `exclusive`: `include/Cellerator/trajectory`
- `read`: `bench`
- `read`: `src/compute/operators`
- `read`: `tests`

## Dependencies
- `checkpoint`: `CE-PTR-SUBSTRATE-CONTRACT-READY`
<!-- todo-orchestrator:v2-managed:end -->
