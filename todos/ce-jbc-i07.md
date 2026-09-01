

<!-- todo-orchestrator:v2-managed:start -->
# CE-JBC-I07: Define operation decomposition alternative ABI v1

Task revision: `3841`; current project revision is in `todo-status.md`.

## Objective
Define operation decomposition alternative ABI v1. Deliver this as one isolated, reviewable step in the Cellerator-owned joint-compiler thin-waist interfaces workstream.

## State
- Lifecycle: `done`
- Execution: `closed`
- Parallel policy: `serial`
- Result: `implemented`

## Next Action
_None._

## Ownership
- `exclusive`: `include/Cellerator/compute/decomposition`
- `exclusive`: `include/Cellerator/execution/joint_compiler`
- `exclusive`: `include/Cellerator/profiling/joint_compiler`
- `exclusive`: `src/execution/joint_compiler`
- `exclusive`: `tests/jbc/interfaces`
- `read`: `include/Cellerator/compute/operation/operation_core_v2/schema.hh`
- `read`: `include/Cellerator/execution/biological_abi.hh`
- `read`: `include/Cellerator/execution/identity.hh`
- `read`: `include/Cellerator/execution/index_space/hierarchical_index_space_v1.hh`
- `read`: `include/Cellerator/execution/lifetimes.hh`
- `read`: `include/Cellerator/profiling/partition_export.h`

## Dependencies
- `task`: `CE-JBC-I06`
<!-- todo-orchestrator:v2-managed:end -->
