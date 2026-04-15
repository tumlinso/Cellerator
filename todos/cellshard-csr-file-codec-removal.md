---
slug: "cellshard-csr-file-codec-removal"
status: "in_progress"
execution: "claimed"
owner: "codex"
created_at: "2026-04-14T19:32:36Z"
last_heartbeat_at: "2026-04-15T13:23:53Z"
last_reviewed_at: "2026-04-15T13:23:53Z"
stale_after_days: 14
objective: "remove CSR/compressed from the CellShard .csh5 file codec, keep CSR only as interop if still needed"
---

# Current Objective

## Summary
Plan the removal of CSR/compressed as a `.csh5` payload family while preserving any needed CSR interop boundary and defining a safe migration path for legacy tests, docs, and API.

## Quick Start
- Why this stream exists: `.csh5` still exposes compressed/standard-CSR create/load/fetch entrypoints even though the repo posture is Blocked-ELL-first for persisted execution and hot-path compute.
- In scope: compressed `.csh5` writer/reader API audit, deciding whether any legacy compressed-file read support survives temporarily, converting remaining tests/docs to native Blocked-ELL expectations, and defining CSR as interop-only if it remains anywhere.
- Out of scope / dependencies: no AnnData-side schema redesign here, no unrelated sparse-kernel optimization work, and no forced migration of existing user files without an explicit compatibility or conversion story.
- Required skills: `todo-orchestrator` for the ledger; no additional repo-local skill discovery produced a better fit for this planning stream.
- Required references: `AGENTS.md`, `extern/CellShard/README.md`, `extern/CellShard/src/disk/csh5.cuh`, `extern/CellShard/src/disk/csh5.cc`, `extern/CellShard/src/sharded/disk.cuh`, `extern/CellShard/src/sharded/sharded_host.cuh`, `tests/cellshard_dataset_h5_test.cu`, `tests/dataset_workbench_runtime_test.cc`, `src/ingest/dataset/dataset_ingest.cuh`, and `src/workbench/dataset_workbench.cc`.

## Planning Notes
- Treat CSR as an interop contract only unless a legacy `.csh5` read path is deliberately retained for migration.
- Separate the decisions about compressed `.csh5` write support, compressed `.csh5` read support, and in-memory CSR interop so one compatibility concern does not keep the whole file codec alive.
- The active ingest path already writes Blocked-ELL, so the remaining risk is public API, tests, legacy file compatibility, and any metadata or browse paths still assuming compressed `.csh5`.

## Assumptions
- New `.csh5` production output should stay Blocked-ELL-native and should not keep a compressed file codec just for historical symmetry.
- If old compressed `.csh5` files still matter, temporary read-only compatibility is safer than keeping compressed write support alive indefinitely.

## Suggested Skills
- `todo-orchestrator` - Track the CSR-to-interop migration as a separate resumable stream so it does not blur with the broader runtime-service or ingest work.

## Useful Reference Files
- `AGENTS.md` - Repo policy already says Blocked-ELL is the native sparse type for persisted execution, staging, and hot-path compute unless a surface is explicitly fallback-only.
- `extern/CellShard/README.md` - Documents the current public compressed fallback language and the canonical-vs-pack runtime split that this change will tighten.
- `extern/CellShard/src/disk/csh5.cuh` - Defines the public compressed `.csh5` create/load/fetch entrypoints and codec-family enum that would need to change.
- `extern/CellShard/src/disk/csh5.cc` - Implements the compressed payload group, materialization path, and cached-pack logic that currently keep CSR alive as a file codec.
- `extern/CellShard/src/sharded/disk.cuh` - Routes `sharded<sparse::compressed>` header loads through the compressed `.csh5` reader.
- `extern/CellShard/src/sharded/sharded_host.cuh` - Contains the compressed fetch wrappers that would need removal or replacement if the file codec goes away.
- `tests/cellshard_dataset_h5_test.cu` - Still carries explicit compressed `.csh5` roundtrip coverage.
- `tests/dataset_workbench_runtime_test.cc` - Still builds a compressed `.csh5` fixture for metadata/runtime summary coverage.
- `src/ingest/dataset/dataset_ingest.cuh` - Shows that active ingest already emits Blocked-ELL `.csh5`, which lowers the risk of removing compressed write support.
- `src/workbench/dataset_workbench.cc` - Still contains standard-CSR size estimation code that should be re-evaluated once CSR is interop-only.

## Plan
- Audit every remaining compressed `.csh5` writer, reader, loader, fetch wrapper, test fixture, and doc reference to classify it as delete, migrate, or temporary compatibility.
- Decide whether legacy compressed `.csh5` files keep a read-only compatibility path for one transition window or become unsupported immediately.
- Convert the remaining repo tests and docs so Blocked-ELL-native `.csh5` is the only forward-looking file story while CSR survives only as an interop boundary if still needed.
- After the compatibility decision, remove or quarantine the compressed `.csh5` API surface and align summary/probe code with the native layout policy.

## Tasks
- [ ] Inventory all compressed/standard-CSR `.csh5` file-codec entrypoints and call sites.
- [ ] Decide whether legacy compressed `.csh5` read support is kept temporarily or removed outright.
- [ ] Replace remaining compressed `.csh5` tests with Blocked-ELL-native fixtures where possible.
- [ ] Tighten README/docs language so CSR is described as interop-only or legacy-compatibility-only, not as a first-class native file format.
- [ ] Remove compressed `.csh5` write support and any public API that is no longer justified once the migration decision is made.

## Blockers
_None recorded yet._

## Progress Notes
- Added a dedicated workstream to decide whether CSR/compressed should disappear from `.csh5` entirely and survive only as an interop layer.
- Started the first-file freeze implementation: public docs/header comments now describe CSR/compressed as a legacy compatibility or interop path, the workbench runtime test no longer creates new compressed .csh5 fixtures, and cellShardDatasetH5Test now uses a Blocked-ELL side-domain append path plus a separate explicit legacy compressed compatibility check.

## Next Actions
- Start by converting the remaining compressed `.csh5` tests and deciding whether legacy compressed-file read support survives as a temporary compatibility layer.
- If compatibility is retained, make it explicitly read-only and time-bounded instead of preserving compressed write support.

## Done Criteria
- The forward-looking CellShard file story no longer treats CSR/compressed as a native `.csh5` payload family.
- Any surviving CSR surface is clearly scoped to interop or legacy read-only compatibility and documented as such.
- Focused tests and docs no longer depend on creating new compressed `.csh5` datasets unless an intentional compatibility test remains.
