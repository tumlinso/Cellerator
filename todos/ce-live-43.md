

<!-- todo-orchestrator:v2-managed:start -->
# CE-LIVE-43: CelleraTorch build, package, and test fan-in

Task revision: `1979`; current project revision is in `todo-status.md`.

## Objective
Integrate the three adapter lanes, preserve the old copy-based CSR exporter as an explicit compatibility and debug path, validate package consumers and Torch-off native builds, and publish one coherent adapter surface.

## State
- Lifecycle: `planned`
- Execution: `ready`
- Parallel policy: `integration_exclusive`
- Result: `-`

## Next Action
_None._

## Ownership
- `exclusive`: `CMakeLists.txt`
- `exclusive`: `components/CelleraTorch/CMakeLists.txt`
- `exclusive`: `components/CelleraTorch/README.md`
- `exclusive`: `components/CelleraTorch/docs/torch_bindings.md`
- `read`: `components/CelleraTorch`
- `read`: `include/Cellerator`

## Dependencies
- `task`: `CE-LIVE-40`
- `task`: `CE-LIVE-41`
- `task`: `CE-LIVE-42`
<!-- todo-orchestrator:v2-managed:end -->
