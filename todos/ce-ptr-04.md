

<!-- todo-orchestrator:v2-managed:start -->
# CE-PTR-04: Geometry construction intermediates

Task revision: `2513`; current project revision is in `todo-status.md`.

## Objective
Replace inappropriate nested builders, generic coordinate streams, repeated reconstruction, route tape or mask ownership, nested layout metrics, and selection scratch with count-scan-fill, flat relations, exact workspaces, and direct image compilation.

## State
- Lifecycle: `done`
- Execution: `closed`
- Parallel policy: `parallel_safe`
- Result: `implemented`

## Next Action
Inspect CE-GEO ownership immediately before claiming and split or defer only overlapping paths; continue independent layout or construction surfaces.

## Ownership
- `exclusive`: `include/Cellerator/geometry/gating.hh`
- `exclusive`: `include/Cellerator/geometry/gating_cuda.cuh`
- `exclusive`: `include/Cellerator/geometry/layout_metrics.hh`
- `exclusive`: `include/Cellerator/geometry/layout_selector.hh`
- `exclusive`: `include/Cellerator/geometry/pack.hh`
- `exclusive`: `src/geometry/gating.cc`
- `exclusive`: `src/geometry/gating_cuda.cu`
- `exclusive`: `src/geometry/layout_metrics.cc`
- `exclusive`: `src/geometry/layout_selector.cc`
- `exclusive`: `src/geometry/pack.cc`
- `read`: `bench`
- `read`: `include/Cellerator/geometry`
- `read`: `src/geometry`
- `read`: `tests`

## Dependencies
- `checkpoint`: `CE-PTR-SUBSTRATE-CONTRACT-READY`
<!-- todo-orchestrator:v2-managed:end -->
