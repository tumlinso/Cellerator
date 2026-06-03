# Cellerator Programming Style Hint

This guide is authoritative for implementation style inside `Cellerator/`.
It complements `AGENTS.md`, `scope.md`, and `optimization.md`; ownership rules
in those files still decide where behavior belongs.

## File Shape

- Organize implementation files by behavior, not by a whole workflow surface.
- Keep preprocessing compute files split around workspace setup, fleet/reduction,
  QC metric dispatch, normalization, feature statistics, fused sparse workflows,
  and session/workflow orchestration.
- Keep workflow and session files context-light. They may load data, stage
  partitions, copy results, report status, and call compute primitives, but they
  must not own reusable CUDA math.
- If a workflow needs GPU reductions, dense adds, metric packets, sparse
  transforms, normalization/stat operators, fleet collectives, or scratch
  mechanics, add the primitive in the owning compute/runtime layer first and
  call it from the workflow.
- Files over roughly 600 lines need an explicit reason to stay together.
- Files over roughly 1000 lines should be split unless generated code,
  third-party constraints, or a tightly coupled external ABI requires otherwise.

## CUDA Helpers

- Private inline device helpers belong in narrowly named `.cuh` files close to
  the kernels that use them.
- Add a short warning comment in those helper headers: prefer the inline helper
  instead of copy-pasting the expression or microkernel.
- Use inline device functions for reusable per-element, per-entry, per-row, or
  warp-local math that does not own a launch boundary.
- Use standalone kernels for launch boundaries, memory traversal ownership,
  scratch ownership, cross-row reductions, layout-specific fused passes, and
  operations whose occupancy or memory traffic must be inspected directly.

## Performance Bias

- Prefer explicit layout-aware primitives over convenience layers in hot paths.
- Bias Volta/V100 paths toward fusion when it removes sparse HBM traversals,
  launch trains, or host-visible staging without creating register spill or
  forced synchronization.
- Keep separate primitive calls available when tests, partial workflows, or
  correctness comparisons need them.
- Keep contiguous metric packets contiguous when they feed NCCL or peer-copy
  reductions.

## Known Monolith Follow-Ups

The preprocessing split is the first cleanup target. Apply the same rule next
to these files or families:

- `src/compute/neighbors/forward_neighbors/forward_neighbors.cu`
- `src/compute/neighbors/forward_neighbors/cuvs_sharded_knn.cu`
- state-reduce and developmental-time model CUDA files
- `include/Cellerator/dist/distributed.cuh`
