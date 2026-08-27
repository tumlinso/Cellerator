

<!-- todo-orchestrator:v2-managed:start -->
# CE-LIVE-34: CPE2 reload and existing opaque CellShard replay

Task revision: `2256`; current project revision is in `todo-status.md`.

## Objective
Build and reload a CPE2 image containing live projections, carry it through existing opaque CPEXEC01 compatibility delivery, activate typed device views, select and prepare a candidate, and execute quantitatively without moving semantics into CellShard.

## State
- Lifecycle: `done`
- Execution: `closed`
- Parallel policy: `parallel_safe`
- Result: `validated`

## Next Action
_None._

## Ownership
- `exclusive`: `bench/ce_live/replay`
- `exclusive`: `docs/CE_LIVE_REPLAY.md`
- `exclusive`: `tests/persistence/ce_live_program_replay_test.cu`
- `read`: `components/CellPack/include/CellPack/persistence`
- `read`: `include/Cellerator/execution/opaque_artifact.hh`

## Dependencies
- `task`: `CE-LIVE-30`
- `task`: `CE-LIVE-31`
<!-- todo-orchestrator:v2-managed:end -->
