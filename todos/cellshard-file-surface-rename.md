---
slug: "cellshard-file-surface-rename"
status: "done"
execution: "closed"
owner: "codex"
created_at: "2026-04-13T18:08:14Z"
last_heartbeat_at: "2026-04-13T19:44:50Z"
last_reviewed_at: "2026-04-13T19:44:50Z"
stale_after_days: 14
objective: "align CellShard filenames with their actual container and packfile responsibilities"
---

# Current Objective

## Summary
Rename and place the misleading CellShard file surfaces so the `.csh5` container backend and the per-part packfile codec live under purpose-aligned canonical paths while preserving compatibility for existing includes during the transition.

## Quick Start
- Why this stream exists: `src/sharded/series_h5.*` is the canonical `.csh5` container and execution-cache backend, while `src/disk/matrix.*` is the per-part packed payload codec used inside `.pack` files; the filenames hide that distinction.
- In scope: rename the implementation files, update internal and documented include paths, and keep compatibility shims where needed so downstream consumers do not break immediately.
- Out of scope / dependencies: no storage-format semantic changes, no public API rename of existing `series_h5` function symbols unless required for correctness, and no broad reorganization beyond these misleading file names.
- Required skills: `todo-orchestrator` for the ledger; no extra repo-local skill is needed unless the rename exposes a CUDA/build-specific break.
- Required references: `extern/CellShard/src/CellShard.hh`, `extern/CellShard/CMakeLists.txt`, `extern/CellShard/README.md`, `extern/CellShard/src/sharded/series_h5.cuh`, `extern/CellShard/src/disk/matrix.cuh`, and callers under `src/` plus `extern/CellShard/tests/`.

## Planning Notes
- Treat this as a file-identity cleanup, not a format or ABI redesign.
- Prefer a staged rename: make the new filenames canonical, then leave forwarding headers at the old public paths so include compatibility survives the transition.
- Keep the symbol-level `series_h5` vocabulary for now because the user asked about filenames specifically and a symbol rename would expand the blast radius without changing behavior.

## Assumptions
- The user wants the mismatch fixed in code now, not just documented.
- Compatibility includes at the old paths are acceptable because they let the repo move to clearer canonical names without forcing an all-at-once downstream migration.

## Suggested Skills
- `todo-orchestrator` - Track the rename as a standalone workstream so it is resumable and does not blur with the broader CellShard release stream.

## Useful Reference Files
- `extern/CellShard/src/CellShard.hh` - Umbrella include that defines the public header surface and will need canonical include path updates.
- `extern/CellShard/CMakeLists.txt` - Build graph for the inspect library source filenames.
- `extern/CellShard/README.md` - Public docs that currently describe the container and packfile responsibilities.
- `extern/CellShard/src/sharded/csh5.cuh` - Canonical `.csh5` container/backend header after the rename.
- `extern/CellShard/src/disk/packfile.cuh` - Canonical packed payload codec header after the rename.
- `extern/CellShard/src/disk/csh5.cuh` - Canonical `.csh5` container/backend header after moving the container surface under `src/disk/`.
- `extern/CellShard/src/disk/packfile.cuh` - Canonical packed payload codec header beside the container backend.
- `extern/CellShard/src/sharded/csh5.cuh` - Compatibility forwarding header for older sharded-path includes.

## Plan
- Choose purpose-aligned replacement filenames for the `.csh5` container surface and the packed-part codec surface.
- Rename the implementation files and update CellShard build rules and internal includes to use the new canonical names.
- Add compatibility forwarding headers at the old public include paths if needed to avoid breaking downstream consumers immediately.
- Update README and any nearby docs/tests that should point at the canonical new file names.
- Build a focused CellShard target to verify the rename did not break the inspect/package surface.

## Tasks
- [x] Choose the new canonical filenames.
- [x] Rename the CellShard source and header files and update direct includes.
- [x] Add or confirm compatibility shims for old include paths.
- [x] Update docs and package-facing references to the new file names.
- [x] Run a focused CellShard build for validation.

## Blockers
_None recorded yet._

## Progress Notes
- Confirmed the mismatch before editing: `series_h5.*` owns the `.csh5` container and shard-pack cache path, while `disk/matrix.*` implements the per-part packed payload codec used by `.pack` files.
- Renamed the canonical container/backend files to `src/sharded/csh5.cuh` and `src/sharded/csh5.cc`, and the packed payload codec files to `src/disk/packfile.cuh` and `src/disk/packfile.cu`.
- Updated the CellShard build graph, umbrella include, internal includes, downstream repo includes, and README to point at the new canonical filenames.
- Added compatibility forwarding headers at `src/sharded/series_h5.cuh` and `src/disk/matrix.cuh` so existing include paths still compile.
- Validated the rename with a fresh standalone configure, `cmake --build /tmp/cellshard-file-surface-rename -j 4 --target cellShardExportRuntimeTest cellShardInspectPackageTest`, and `./cellShardExportRuntimeTest`.
- Resumed the file-surface rename stream to test whether the  container backend belongs under  rather than .
- Moved the canonical `.csh5` container/backend files from `src/sharded/` to `src/disk/csh5.cuh` and `src/disk/csh5.cc` so they sit beside the on-disk `packfile` codec.
- Updated the CellShard build graph, umbrella include, internal includes, downstream repo includes, and README to use `disk/csh5.*` as the canonical path.
- Kept `src/sharded/csh5.cuh` and `src/sharded/series_h5.cuh` as forwarding compatibility headers so older include paths still compile.
- Revalidated the layout with `cmake --build /tmp/cellshard-file-surface-rename -j 4 --target cellShardExportRuntimeTest cellShardInspectPackageTest` and `./cellShardExportRuntimeTest` from `/tmp/cellshard-file-surface-rename`.
- Confirmed there were no remaining references to `src/sharded/csh5.cuh`, `src/sharded/series_h5.cuh`, or `src/disk/matrix.cuh`, then deleted those compatibility headers.
- Revalidated the post-shim surface with `cmake --build /tmp/cellshard-file-surface-rename -j 4 --target cellShardExportRuntimeTest cellShardInspectPackageTest` and `./cellShardExportRuntimeTest` from `/tmp/cellshard-file-surface-rename`.

## Next Actions
- No further action in this stream unless the user wants a follow-up symbol/API vocabulary rename beyond the file-surface cleanup.
- Update includes/build/docs for , keep compatibility forwarding headers, and rerun the focused CellShard validation.
- No further action in this stream unless the user wants `sharded/disk.cuh` renamed or the remaining `series_h5` symbol vocabulary changed.
- No further action in this stream unless the user wants additional naming cleanup such as `sharded/disk.cuh` or the remaining `series_h5` symbol vocabulary.

## Done Criteria
- The canonical CellShard sources use purpose-aligned filenames for the `.csh5` container and packfile codec responsibilities.
- Old include paths continue to compile through explicit compatibility forwarding headers or another intentional compatibility mechanism.
- CellShard build/test documentation points at the new canonical file names.
- A focused build succeeds after the rename.
