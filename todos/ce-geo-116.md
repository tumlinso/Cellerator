

<!-- todo-orchestrator:v2-managed:start -->
# CE-GEO-116: Real-data tileability census

Task revision: `3281`; current project revision is in `todo-status.md`.

## Objective
Census PBMC3K negative control, available developmental embryo data, at least one heart-relevant relation/dataset, controlled synthetic structures, and checked perturbation/multiome/regulatory/trajectory manifests without core parsing or storage ownership.

## State
- Lifecycle: `done`
- Execution: `closed`
- Parallel policy: `parallel_safe`
- Result: `evaluated_not_promoted`

## Next Action
_None._

## Ownership
- `exclusive`: `bench/biology/ce_geo/manifests`
- `exclusive`: `bench/biology/ce_geo/tileability_census.cc`
- `exclusive`: `bench/ce_geo/evidence/biology/tileability.jsonl`
- `read`: `bench/architecture_evidence/real_traces`
- `read`: `bench/ce_geo/harness`
- `read`: `bench/real_data`
- `read`: `data/manifests`

## Dependencies
- `checkpoint`: `CE-GEO-RECTANGULAR-SUPPORT-V1`
- `checkpoint`: `CE-GEO-BENCH-INFRA-V1`
<!-- todo-orchestrator:v2-managed:end -->
