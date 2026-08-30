

<!-- todo-orchestrator:v2-managed:start -->
# CE-GEO-101: Foundation ABI and provider negative tests

Task revision: `3016`; current project revision is in `todo-status.md`.

## Objective
Test invalid and duplicate providers/candidates, stale capability, wrong device, incompatible numeric tuple, failed atomic assembly, and no query after sealing.

## State
- Lifecycle: `done`
- Execution: `closed`
- Parallel policy: `serial`
- Result: `validated`

## Next Action
_None._

## Ownership
- `exclusive`: `tests/ce_geo/validation/foundation_negative_test.cu`
- `read`: `include/Cellerator/compute/architecture`
- `read`: `include/Cellerator/compute/operation/candidate_catalog_v2.hh`
- `read`: `include/Cellerator/runtime/device_descriptor.hh`

## Dependencies
- `checkpoint`: `CE-GEO-FOUNDATION-INTEGRATED`
<!-- todo-orchestrator:v2-managed:end -->
