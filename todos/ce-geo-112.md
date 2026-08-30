

<!-- todo-orchestrator:v2-managed:start -->
# CE-GEO-112: N=64 output-owned kernel calibration

Task revision: `3016`; current project revision is in `todo-status.md`.

## Objective
Compare historical experiment, output-owned N64 WMMA, best local sparse baselines, and legal dense baselines with mechanism and numerical evidence.

## State
- Lifecycle: `done`
- Execution: `closed`
- Parallel policy: `serial`
- Result: `evaluated_not_promoted`

## Next Action
_None._

## Ownership
- `exclusive`: `bench/ce_geo/evidence/micro/n64_output_owned.jsonl`
- `exclusive`: `bench/tensor_core/ce_geo/n64_output_owned.cu`
- `read`: `src/compute/architecture/providers/nvidia/sm70/relation_apply_n64.cu`
- `read`: `src/compute/candidate/csr_fallback_candidate.cu`
- `read`: `src/compute/candidate/tensor_core/v100_dense_fragment_candidate.cu`

## Dependencies
- `task`: `CE-GEO-73`
<!-- todo-orchestrator:v2-managed:end -->
