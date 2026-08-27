

<!-- todo-orchestrator:v2-managed:start -->
# CE-LIVE-41: CelleraTorch forward custom operation wrapper

Task revision: `2043`; current project revision is in `todo-status.md`.

## Objective
Wrap executable_program as a Torch custom operation using the current Torch CUDA stream, preserving Cellerator planning and ownership and performing no hidden repeated conversion.

## State
- Lifecycle: `planned`
- Execution: `ready`
- Parallel policy: `parallel_safe`
- Result: `-`

## Next Action
_None._

## Ownership
- `exclusive`: `components/CelleraTorch/docs/program_ops.md`
- `exclusive`: `components/CelleraTorch/include/CelleraTorch/program_ops.hh`
- `exclusive`: `components/CelleraTorch/src/program_ops.cu`
- `exclusive`: `components/CelleraTorch/tests/program_ops_test.cc`
- `read`: `include/Cellerator/execution/program.hh`

## Dependencies
- `task`: `CE-LIVE-37`
<!-- todo-orchestrator:v2-managed:end -->
