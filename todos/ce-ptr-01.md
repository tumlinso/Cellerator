

<!-- todo-orchestrator:v2-managed:start -->
# CE-PTR-01: Baseline inventory, policy, and permanent gates

Task revision: `2376`; current project revision is in `todo-status.md`.

## Objective
Create the authoritative live production-core inventory, explicit allowlist policy, permanent source enforcement, and before-migration CPU/GPU allocation, memory, transfer, synchronization, latency, and kernel baseline evidence.

## State
- Lifecycle: `planned`
- Execution: `ready`
- Parallel policy: `parallel_safe`
- Result: `-`

## Next Action
Inspect live code and existing gates; publish CE-PTR-POLICY-READY once classification and enforcement semantics are stable so CE-PTR-02 may proceed while remaining baseline measurements finish.

## Ownership
- `exclusive`: `bench/ce_ptr/baseline`
- `exclusive`: `docs/CE_PTR_INVENTORY.md`
- `exclusive`: `docs/CE_PTR_POLICY.md`
- `exclusive`: `scripts/check_no_inappropriate_core_stl.py`
- `read`: `CMakeLists.txt`
- `read`: `bench`
- `read`: `compat`
- `read`: `components/CelleraTorch`
- `read`: `include/Cellerator`
- `read`: `src`
- `read`: `tests`

## Dependencies
- `checkpoint`: `CE-PTR-BOOTSTRAP-READY`
<!-- todo-orchestrator:v2-managed:end -->
