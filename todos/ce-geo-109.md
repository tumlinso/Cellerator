

<!-- todo-orchestrator:v2-managed:start -->
# CE-GEO-109: Full Volta regression and acceptance evidence

Task revision: `2379`; current project revision is in `todo-status.md`.

## Objective
Run normal and relevant compatibility builds plus all compatibility, structure, runtime, numerical, static, sanitizer, and integrated Volta tests; emit machine-readable source/hardware/command/contamination evidence.

## State
- Lifecycle: `planned`
- Execution: `ready`
- Parallel policy: `serial`
- Result: `-`

## Next Action
_None._

## Ownership
- `exclusive`: `bench/ce_geo/evidence/full_volta_validation.json`
- `exclusive`: `tests/ce_geo/validation/run_full_volta_acceptance.py`
- `read`: `CMakeLists.txt`
- `read`: `bench`
- `read`: `components/CelleraTorch`
- `read`: `include`
- `read`: `src`
- `read`: `tests`

## Dependencies
- `checkpoint`: `CE-GEO-VOLTA-SYSTEM-INTEGRATED`
<!-- todo-orchestrator:v2-managed:end -->
