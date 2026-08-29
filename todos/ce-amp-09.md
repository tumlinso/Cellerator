

<!-- todo-orchestrator:v2-managed:start -->
# CE-AMP-09: Ampere integration and report

Task revision: `2379`; current project revision is in `todo-status.md`.

## Objective
Integrate only validated sm_86 provider, projection, operations, dispositions, build/export wiring, and report; prohibit 2:4 edge pruning and keep TF32/structured sparsity optional and empirical.

## State
- Lifecycle: `planned`
- Execution: `ready`
- Parallel policy: `integration_exclusive`
- Result: `-`

## Next Action
_None._

## Ownership
- `exclusive`: `bench/ce_geo/evidence/ampere/report.md`
- `exclusive`: `src/compute/architecture/providers/nvidia/sm86/CMakeLists.txt`
- `exclusive`: `src/compute/architecture/providers/nvidia/sm86/catalog_fragment.cc`
- `exclusive`: `tests/tensor_core/sm86/CMakeLists.txt`
- `read`: `CMakeLists.txt`
- `read`: `bench/ce_geo/evidence/ampere`
- `read`: `src/compute/CMakeLists.txt`

## Dependencies
- `checkpoint`: `CE-GEO-COMPLETE`
- `decision`: `CE-AMP-PERMISSION`
- `task`: `CE-AMP-02`
- `task`: `CE-AMP-05`
- `task`: `CE-AMP-08`
<!-- todo-orchestrator:v2-managed:end -->
