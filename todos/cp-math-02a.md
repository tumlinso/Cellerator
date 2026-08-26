<!-- todo-orchestrator:v2-managed:start -->
# CP-MATH-02A: Device context, capabilities, fingerprint, and workspace pool

Task revision: `1418`; current project revision is in `todo-status.md`.

## Objective
Independently implement cached DeviceMathContext, DeviceCapabilities, DeviceFingerprint, and reusable workspace ownership over the existing runtime context/handle/scratch substrate without defining operation-dependent backend policy.

## State
- Lifecycle: `superseded`
- Execution: `closed`
- Parallel policy: `serial`
- Result: `superseded`

## Next Action
_None._

## Ownership
- `exclusive`: `Cellerator/include/Cellerator/compute/math/runtime.hh`
- `exclusive`: `Cellerator/src/compute/math/runtime`
- `exclusive`: `Cellerator/tests/math_device_runtime_test.cu`
- `forbidden`: `Cellerator/components/CellPack`
- `read`: `Cellerator/include/Cellerator/runtime`
- `read`: `Cellerator/src/runtime/runtime.cu`

## Dependencies
_None._
<!-- todo-orchestrator:v2-managed:end -->
