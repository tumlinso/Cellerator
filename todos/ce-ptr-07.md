

<!-- todo-orchestrator:v2-managed:start -->
# CE-PTR-07: Sampling and sampled structural ownership

Task revision: `2510`; current project revision is in `todo-status.md`.

## Objective
Migrate sampling plans, results, and sampled CSR construction to explicit images and workspaces while preserving deterministic reproduction, provenance, stable row identity, and Cellerator's non-storage boundary.

## State
- Lifecycle: `done`
- Execution: `closed`
- Parallel policy: `parallel_safe`
- Result: `implemented`

## Next Action
Define explicit sample image and workspace contracts, evaluate redundant hashes and weights from live provenance, and preserve exact reproduction.

## Ownership
- `exclusive`: `include/Cellerator/compute/sampling.hh`
- `exclusive`: `include/Cellerator/compute/sampling_materialization.hh`
- `exclusive`: `src/compute/dataset/sampling.cc`
- `exclusive`: `src/compute/dataset/sampling_materialization.cc`
- `read`: `bench`
- `read`: `include/Cellerator/geometry`
- `read`: `src/geometry`
- `read`: `tests`

## Dependencies
- `checkpoint`: `CE-PTR-SUBSTRATE-CONTRACT-READY`
<!-- todo-orchestrator:v2-managed:end -->
