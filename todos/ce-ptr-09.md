

<!-- todo-orchestrator:v2-managed:start -->
# CE-PTR-09: Device-resident support to candidate to merge pipeline

Task revision: `2506`; current project revision is in `todo-status.md`.

## Objective
Redesign gene support, candidate discovery, and exact merge scoring as one device-resident prepared pipeline with explicit views and workspaces, preflight capacities and CUB scratch, and terminal-only host materialization.

## State
- Lifecycle: `done`
- Execution: `closed`
- Parallel policy: `parallel_safe`
- Result: `implemented`

## Next Action
Treat the three stages as one pipeline, retain authoritative host referees, and collect before-after V100 end-to-end evidence under the benchmark mutex.

## Ownership
- `exclusive`: `include/Cellerator/geometry/gene_candidate_discovery.hh`
- `exclusive`: `include/Cellerator/geometry/gene_support_bitset.hh`
- `exclusive`: `include/Cellerator/geometry/merge_cost.hh`
- `exclusive`: `src/compute/dataset/gene_support_bitset.cu`
- `exclusive`: `src/geometry/candidate_discovery`
- `exclusive`: `src/geometry/merge_cost.cc`
- `exclusive`: `src/geometry/merge_cost_cuda.cu`
- `read`: `bench`
- `read`: `include/Cellerator/runtime`
- `read`: `src/runtime`
- `read`: `tests`

## Dependencies
- `checkpoint`: `CE-PTR-SUBSTRATE-CONTRACT-READY`
<!-- todo-orchestrator:v2-managed:end -->
