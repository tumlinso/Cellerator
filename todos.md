# Active Objectives

## Summary
Clean up stale repo documentation, remove obsolete planning notes, and align the source tree and durable references with the code that actually exists.

## Shared Assumptions
- `README.md` remains the main entrypoint for repo orientation.
- `todos.md` remains the only active planning ledger unless a future task needs new workstream files.
- `docs/` is out of scope for this cleanup.
- Functional code behavior should not change as part of this pass.

## Suggested Skills
- `todo-orchestrator` - Keep the root ledger current for substantial cleanup work.

## Useful Reference Files
- `AGENTS.md` - Repo-specific engineering rules and path expectations.
- `README.md` - Main repo overview that should match the real tree.
- `optimization.md` - Durable performance guidance that should stay intact.
- `pointer_migration_plan.md` - Durable hot-path migration policy.
- `custom_torch_ops.md` - Durable model-op registry.

## Workstreams
_No active workstream files._

## Global Blockers
_None recorded._

## Progress Notes
- Collapsed stale worker planning into root `todos.md`.
- Removed obsolete top-level planning notes and historical scratch docs.
- Renamed the workbench directory to match its actual role without changing behavior.
- Rewrote root and module docs to describe only active code and durable references.

## Historical Summary
- Recent completed work included Blocked-ELL persistence, real-data sparse replay benchmarking, quantize autograd kernels, and workbench browse-cache updates.
- Detailed historical workstream notes are preserved in git history rather than as active repo ledgers.

## Next Actions
- Keep new planning in root `todos.md` until another task genuinely needs concurrent workstreams.
- Treat stale one-off notes as disposable unless they are promoted into a durable reference document.

## Done Criteria
- The repo has one active planning ledger.
- Root docs match the real directory layout.
- Stale planning scratch files are removed.
- No references remain to deleted docs or the old workbench path.
