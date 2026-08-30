

<!-- todo-orchestrator:v2-managed:start -->
# CE-GEO-88: Relation-algebra interface freeze

Task revision: `3016`; current project revision is in `todo-status.md`.

## Objective
Prove exact semantics, examples using only public Cellerator APIs, compatibility, segments, bundles, and absence of model/trainer/autograd ownership.

## State
- Lifecycle: `done`
- Execution: `closed`
- Parallel policy: `serial`
- Result: `implemented`

## Next Action
_None._

## Ownership
- `exclusive`: `tests/relation_algebra/interface_test.cu`
- `read`: `examples/ce_geo`
- `read`: `include/Cellerator/compute/operation/relation_algebra.hh`
- `read`: `src/compute/candidate/relation_bundle.cc`
- `read`: `src/compute/candidate/segment`

## Dependencies
_None._
<!-- todo-orchestrator:v2-managed:end -->
