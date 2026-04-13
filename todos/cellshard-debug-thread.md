---
slug: "cellshard-debug-thread"
status: "stale"
execution: "closed"
owner: "unassigned"
created_at: "2026-04-13T14:45:32Z"
last_heartbeat_at: "2026-04-13T14:48:10Z"
last_reviewed_at: "2026-04-13T14:48:10Z"
stale_after_days: 14
objective: "maintain a pickup ready CellShard debugging thread for crash triage and regression isolation"
stale_reason: "User marked this stream stale; keep the debug ledger for history but do not pick it up unless a new CellShard crash reactivates it."
---

# Current Objective

## Summary
Keep a narrow CellShard debug stream that records current crashes, repro commands, likely causes, and the next concrete debug step so another thread can pick it up directly.

## Quick Start
- Why this stream exists: CellShard now has enough independent surface area that crash triage and runtime regression work should not be mixed into the deep terminology rename ledger.
- In scope: focused crash triage, reproducer capture, likely-cause notes, and regression-test planning for CellShard runtime failures.
- Out of scope and dependencies: broad feature work and the core partition rename belong in their own streams unless a debug result directly unblocks them.
- Required skills: todo-orchestrator for maintaining the pickup context and cuda-v100 only if a bug turns out to depend on GPU execution behavior.
- Required references: AGENTS.md, extern/CellShard/src/sharded/series_h5.cc, extern/CellShard/src/sharded/series_h5.cuh, src/workbench/series_workbench.cc, tests/series_workbench_runtime_test.cc, and any current crash note from this ledger.

## Planning Notes
- The first debug target in this stream is the current seriesWorkbenchRuntimeTest crash, which backtraces into cellshard create_series_compressed_h5 before the renamed summary layer runs.
- This stream should keep the distinction between rename-induced regressions and pre-existing unrelated crashes explicit.

## Assumptions
- Treat the current seriesWorkbenchRuntimeTest crash as unrelated to the public summary rename until evidence shows otherwise, because the observed backtrace lands inside create_series_compressed_h5 before the renamed export or workbench summary code executes.
- Every debug pickup should leave behind a short repro command, current hypothesis, and next concrete step rather than a generic note.

## Suggested Skills
- `todo-orchestrator` - Keep the debug stream legible and pickup ready for future crash work.
- `cuda-v100` - Use only if a CellShard debug issue turns into a GPU runtime or sparse execution investigation.

## Useful Reference Files
- `AGENTS.md` - Repo-level guardrails for testing and performance-sensitive edits.
- `extern/CellShard/src/sharded/series_h5.cc` - Current crash backtrace lands in create_series_compressed_h5 here.
- `extern/CellShard/src/sharded/series_h5.cuh` - Public series HDF5 layout declarations and constants.
- `src/workbench/series_workbench.cc` - Caller path that triggers the current crashing test setup.
- `tests/series_workbench_runtime_test.cc` - Current reproducer for the active CellShard crash.

## Plan
- Reproduce the active CellShard crash with a stable command and capture the exact failing function and setup path.
- Reduce the failure to the smallest useful local setup and identify the first invalid pointer, layout field, or metadata contract violation.
- Decide whether the failure predates the deep partition rename or is a new regression from recent CellShard work.
- Patch the root cause and add a focused regression test or assertion if the failure is deterministic.

## Tasks
- [x] Reproduce the current seriesWorkbenchRuntimeTest segfault and keep the exact command in this ledger.
- [x] Inspect the create_series_compressed_h5 call path and input layout objects used by the failing test.
- [x] Determine whether the crash predates the deep partition rename or is caused by another recent CellShard change.
- [x] Add a focused regression check once the root cause is fixed.

## Blockers
_None recorded yet._

## Progress Notes
- Observed state at handoff: seriesWorkbenchRuntimeTest currently crashes with SIGSEGV, and cuda-gdb shows the fault inside cellshard create_series_compressed_h5.
- Likely cause from the dedicated debug thread: tests/series_workbench_runtime_test.cc declares series_layout_view without zero-initialization and never sets layout.part_aux before calling create_series_compressed_h5, so the compressed writer may dereference a garbage non-null part_aux pointer.
- Current working hypothesis: this crash predates the public partition rename because the fault happens in the low-level HDF5 writer before the renamed summary layer executes.
- Focused debug note from the side thread: the current seriesWorkbenchRuntimeTest crash is likely an uninitialized series_layout_view.part_aux in the test fixture, not the public partition rename.
- Patched the failing test fixture to zero-initialize the series HDF5 view structs and set layout.part_aux = nullptr before create_series_compressed_h5.
- Found the same uninitialized compressed-series fixture pattern in tests/cellshard_series_h5_test.cu and patched it to zero-initialize the view structs and set layout.part_aux = nullptr.
- Confirmed the crash fix: zero-initializing the series HDF5 view structs and setting layout.part_aux = nullptr removed the create_series_compressed_h5 segfault in tests/series_workbench_runtime_test.cc.
- Patched the same latent fixture bug in tests/cellshard_series_h5_test.cu so the compressed-series HDF5 tests do not rely on uninitialized view state.
- Validation after the fix: /tmp/cellerator_cellshard_split_build/seriesWorkbenchRuntimeTest returned 0 and /tmp/cellerator_cellshard_split_build/cellShardSeriesH5Test returned 0 when run serially.
- One intermediate parallel rerun produced HDF5 file-lock noise; the real validation path for these tests should stay serial.

## Next Actions
- Read the current crash note, run the reproducer, and inspect the create_series_compressed_h5 inputs before changing code.
- Zero-initialize the series_layout_view fixture or set layout.part_aux to nullptr in the failing test, then rerun seriesWorkbenchRuntimeTest to confirm the hypothesis.
- Keep the first debug patch narrow and add a focused assertion or regression test if the root cause is confirmed.
- Run the native-debugging crash capture for /tmp/cellerator_cellshard_split_build/seriesWorkbenchRuntimeTest and inspect the summary files first.
- Rebuild seriesWorkbenchRuntimeTest and rerun the exact reproducer to confirm the fixture hypothesis.
- Rebuild and run seriesWorkbenchRuntimeTest and cellShardSeriesH5Test serially to confirm the fixture-family fix.
- For the next CellShard issue, run native-debugging/scripts/debug_crash.sh first and read summary.txt before choosing a follow-on tool.
- Keep CellShard runtime-test reruns serial when they touch the same temporary HDF5 files.

## Done Criteria
- The current CellShard crash has a stable repro, an identified cause, and a recorded fix or explicit blocker.
- The debug stream contains enough context that another thread can resume the next CellShard triage step without rediscovery.
- Any fixed CellShard runtime crash covered by this stream has a focused regression test or assertion.
