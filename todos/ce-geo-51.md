

<!-- todo-orchestrator:v2-managed:start -->
# CE-GEO-51: Deterministic sampled support extraction

Task revision: `3281`; current project revision is in `todo-status.md`.

## Objective
Reuse compatible sampled-support machinery, add bounded high-degree pair sampling, deterministic provenance and seed, and avoid all-pairs expansion.

## State
- Lifecycle: `done`
- Execution: `closed`
- Parallel policy: `serial`
- Result: `implemented`

## Next Action
_None._

## Ownership
- `exclusive`: `src/geometry/support_atlas.cc`
- `exclusive`: `tests/geometry/ce_geo/sampled_support_test.cc`
- `read`: `include/Cellerator/compute/sampling.hh`
- `read`: `include/Cellerator/geometry/gene_support_bitset.hh`
- `read`: `src/compute/dataset/sampling.cc`

## Dependencies
_None._
<!-- todo-orchestrator:v2-managed:end -->
