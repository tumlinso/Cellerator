# Active Objectives

## Summary
Use this file as the canonical index for substantial multi-step work.

## Shared Assumptions
_None recorded yet._

## Suggested Skills
- `todo-orchestrator` - Track the release plan and keep audit findings visible alongside active implementation work.
- `todo-orchestrator` - Track the ingest/runtime migration and keep the execution ledger current.
- `cuda-v100` - Use only if Blocked-ELL runtime/preprocess kernel behavior or V100 fit becomes the limiting issue.
- `todo-orchestrator` - Track the rename as a standalone workstream so it is resumable and does not blur with the broader CellShard release stream.
- `todo-orchestrator` - Track the architecture reset as a standalone resumable stream.
- `cuda-v100` - Use only if the runtime delivery or staging changes expose new Volta bottlenecks.
- `todo-orchestrator` - Track the CSR-to-interop migration as a separate resumable stream so it does not blur with the broader runtime-service or ingest work.

## Useful Reference Files
- `extern/CellShard/README.md` - Current stated scope of the standalone CellShard library.
- `extern/CellShard/CMakeLists.txt` - Defines the package exports, tests, install surface, and current version file.
- `extern/CellShard/pyproject.toml` - Defines the optional Python release surface and currently claims version 0.1.0.
- `extern/CellShard/src/CellShard.hh` - Umbrella include showing the intended public entry surface.
- `extern/CellShard/src/sharded/sharded.cuh` - Canonical partition and shard naming used by the current API.
- `src/models/dense_reduce/dR_dataloader.hh` - Concrete downstream code still using the pre-rename `part` API names.
- `src/models/dense_reduce/dR_infer.hh` - Additional downstream breakage against the renamed CellShard surface.
- `extern/CellShard/src/sharded/csh5.cuh` - Public csh5 and cache-management surface.
- `extern/CellShard/src/sharded/csh5.cc` - Actual csh5 writer, cache pack builder, and shard fetch behavior.
- `extern/CellShard/src/sharded/disk.cuh` - Type-directed `.csh5` header loading entrypoints.
- `src/ingest/series/series_ingest.cuh` - Current MTX-to-csh5 writer path that always emits Blocked-ELL.
- `src/workbench/series_workbench.cc` - Current ingest planner and shard/execution metadata logic.
- `src/workbench/series_workbench_cuda.cu` - Current conversion driver, browse-cache builder, and preprocess runtime assumptions.
- `extern/CellShard/src/CellShard.hh` - Umbrella include that defines the public header surface and will need canonical include path updates.
- `extern/CellShard/CMakeLists.txt` - Build graph for the inspect library source filenames.
- `extern/CellShard/README.md` - Public docs that currently describe the container and packfile responsibilities.
- `extern/CellShard/src/sharded/series_h5.cuh` - Current misplaced filename for the `.csh5` container backend.
- `extern/CellShard/src/disk/matrix.cuh` - Current misplaced filename for the per-part packed payload codec.
- `extern/CellShard/README.md` - Current public ownership split and csh5/pack wording that needs to be tightened.
- `extern/CellShard/src/disk/csh5.cuh` - Public csh5 schema structs and append/load APIs.
- `extern/CellShard/src/disk/csh5.cc` - Current execution metadata persistence, cache runtime, and pack materialization code.
- `tests/cellshard_series_h5_test.cu` - Focused roundtrip/runtime test surface for new csh5 metadata.
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

## Workstreams
- `dual-cuda-optimization-modes` | status: in_progress | owner: codex | file: `todos/dual-cuda-optimization-modes.md` | objective: add repo-wide generic, native, and native-extreme CUDA modes with explicit topology policy and first hotspot backends
- `cellshard-first-stable-release` | status: in_progress | owner: codex | file: `todos/cellshard-first-stable-release.md` | objective: bring CellShard to a first stable release
- `cellshard-blocked-ell-ingest-runtime` | status: in_progress | owner: codex | file: `todos/cellshard-blocked-ell-ingest-runtime.md` | objective: implement blocked-ell-first CellShard ingest, explicit machine-local cache warmup, and runtime alignment
- `cellshard-file-surface-rename` | status: done | owner: codex | file: `todos/cellshard-file-surface-rename.md` | objective: align CellShard filenames with their actual container and packfile responsibilities
- `cellshard-runtime-service-contract` | status: in_progress | owner: codex | file: `todos/cellshard-runtime-service-contract.md` | objective: reset CellShard around owner-hosted pack delivery, append-only canonical generations, and a Cellerator immutable-emission boundary
- `cellshard-csr-file-codec-removal` | status: in_progress | owner: codex | file: `todos/cellshard-csr-file-codec-removal.md` | objective: remove CSR/compressed from the CellShard .csh5 file codec, keep CSR only as interop if still needed

## Global Blockers
_None recorded yet._

## Progress Notes
- Ran `todo-cleanup --partial` and cleared workstreams: cellshard-core-partition-rename.
- Configured a standalone audit build with `cmake -S extern/CellShard -B /tmp/cellshard-release-audit -DCELLSHARD_BUILD_TESTS=ON -DCELLSHARD_ENABLE_PYTHON=OFF -DCELLSHARD_BUILD_APPS=OFF`.
- Built `cellShardExportRuntimeTest` successfully and ran it successfully in the audit build.
- Built the staged install/package-consumer path and confirmed `cellShardInspectPackageTest` fails when the consumer includes `CellShard/src/CellShard.hh` against exported include dirs rooted at `${prefix}/include/CellShard`.
- Built the root `modelCustomOpsTest` path and confirmed the current Cellerator downstream breakage is the `part` to `partition` API migration, not a packaging-only issue.
- Tried `python3 -m pip wheel extern/CellShard -w /tmp/cellshard-dist --no-deps` and observed the optional Python wheel fail while linking `cellshardH5adExport`.
- Audited the standalone CellShard release surface and added the planned workstream `cellshard-first-stable-release` with concrete package, downstream, and Python packaging blockers.
- Claimed the release workstream for implementation; starting with install/package contract and Python linkage.
- Patched the standalone CellShard package contract so `cellShardInspectPackageTest` now validates both an inspect-only C++ consumer and a CUDA runtime consumer against the installed package.
- Linked `cellshard_inspect` transitively against `CUDA::cudart` in CUDA builds and passed the parent CUDA toolchain settings into the nested package-consumer configure step.
- Hardened Python packaging by removing the wheel-time dependency on the standalone `cellshardH5adExport` app, adding Python package classifiers, and exposing `cellshard.__version__`.
- Validated both `python3 -m pip wheel extern/CellShard -w /tmp/cellshard-dist --no-deps` and `python3 -m venv /tmp/cellshard-src-venv && /tmp/cellshard-src-venv/bin/pip install extern/CellShard` with successful `import cellshard` smoke checks.
- Migrated `src/models/dense_reduce/` to the current CellShard partition-named API and rebuilt `modelCustomOpsTest` successfully.
- Added release-facing basics inside `extern/CellShard`: `CHANGELOG.md`, `SUPPORT.md`, README support/package notes, and a starter CI workflow covering hosted CPU packaging plus self-hosted CUDA validation.
- Added  under Apache 2.0 and aligned the Python package metadata to ; reran the wheel build successfully.
- Added `extern/CellShard/LICENSE` under Apache 2.0 and aligned the Python package metadata to `license = "Apache-2.0"`; reran the wheel build successfully.
- Created the ingest/runtime migration workstream and recorded the implementation order: runtime alignment, cache warmup, shard-sizing split, writer alignment, then tests.
- Implementation started for bucketed Blocked-ELL execution packs driven by .csh5 execution metadata.
- Retargeted the active CellShard backend work to `extern/CellShard/src/sharded/csh5.*`, finished the first bucketed execution-pack integration slice, and restored green runtime coverage with `./build/seriesWorkbenchRuntimeTest` plus `./build/cellShardSeriesH5Test`.
- Confirmed the mismatch before editing: `series_h5.*` owns the `.csh5` container and shard-pack cache path, while `disk/matrix.*` implements the per-part packed payload codec used by `.pack` files.
- Added a dedicated workstream for the CellShard file-surface rename so the naming cleanup is tracked separately from the broader release/runtime work.
- Completed the CellShard file-surface rename workstream: canonical filenames now reflect the `.csh5` container and `.pack` codec roles, with compatibility headers preserving old include paths.
- Renamed the canonical container/backend files to `src/sharded/csh5.cuh` and `src/sharded/csh5.cc`, and the packed payload codec files to `src/disk/packfile.cuh` and `src/disk/packfile.cu`.
- Updated the CellShard build graph, umbrella include, internal includes, downstream repo includes, and README to point at the new canonical filenames.
- Added compatibility forwarding headers at `src/sharded/series_h5.cuh` and `src/disk/matrix.cuh` so existing include paths still compile.
- Validated the rename with a fresh standalone configure, `cmake --build /tmp/cellshard-file-surface-rename -j 4 --target cellShardExportRuntimeTest cellShardInspectPackageTest`, and `./cellShardExportRuntimeTest`.
- Resumed the file-surface rename stream to test whether the  container backend belongs under  rather than .
- Refined the CellShard file-surface rename: the canonical `.csh5` container backend now lives under `extern/CellShard/src/disk/` beside the packfile codec, while `sharded/` keeps compatibility-only forwarding headers.
- Removed the last CellShard compatibility headers for the old file paths after confirming there were no remaining include-site dependencies.
- Confirmed there were no remaining references to `src/sharded/csh5.cuh`, `src/sharded/series_h5.cuh`, or `src/disk/matrix.cuh`, then deleted those compatibility headers.
- Revalidated the post-shim surface with `cmake --build /tmp/cellshard-file-surface-rename -j 4 --target cellShardExportRuntimeTest cellShardInspectPackageTest` and `./cellShardExportRuntimeTest` from `/tmp/cellshard-file-surface-rename`.
- Landed the next CellShard Blocked-ELL ingest/runtime slice: `.csh5` now persists shard-scoped optimized Blocked-ELL payloads with shard-local column maps, ingest emits those optimized shards directly, and the blocked-ell workbench runtime remaps shard-local columns correctly in browse/preprocess hot paths.
- Created a dedicated workstream for the CellShard runtime-service contract reset so it does not blur with the older blocked-ell ingest/runtime stream.
- Extended `extern/CellShard/src/disk/csh5.cuh` and `csh5.cc` with optional runtime-service metadata, generation identity fields, shard owner node/rank arrays, and inspect accessors for loaded series backends.
- Updated the workbench writer to append default owner-hosted runtime-service metadata when it records execution metadata for a new series.
- Rewrote the CellShard and Cellerator ingest/runtime docs so pack is described as the primary execution artifact and Cellerator is limited to immutable canonical emission before CellShard layout/build work.
- Built `cellShardSeriesH5Test` successfully and ran `./build/cellShardSeriesH5Test` successfully with the new runtime-service metadata roundtrip coverage.
- Expanded the docs to spell out the owner-hosted runtime model in single-machine and distributed operation, including coordinator, master reader, pack-prep, executor, and append-staging roles.
- Reworked the CellShard cache tree into `instances/<fingerprint>/metadata`, `packs/canonical`, and `packs/execution`, wrote the new layout into the cache manifest, and restored green `cellShardSeriesH5Test` coverage after fixing the canonical execution-pack column-map serialization path.
- Tightened the first generated `.csh5` validation path: workbench summaries now expose `matrix_format`, `payload_layout`, execution `preferred_base_format`, and runtime-service metadata, and `seriesWorkbenchRuntimeTest` now reopens the converted file to assert the persisted optimized blocked-ell codec and non-identity shard-local column remap.
- Added a dedicated workstream to decide whether CSR/compressed should disappear from `.csh5` entirely and survive only as an interop layer.
- Started the approved rename from `portable` / `extreme` to `generic` / `native` / `native-extreme` in the root and nested CellShard CMake/config surfaces.
- Current implementation focus is the topology contract rewrite: make generic multi-GPU behavior discover-and-adapt from peer connectivity instead of assuming the native `0<->2` and `1<->3` pairs in autograd and forward-neighbor code.
- Completed the mode rename to `generic` / `native` / `native-extreme`, updated the generated mode header and nested CellShard compile definitions, and switched the quantized native-PTX selection checks to the new `native-extreme` flag.
- Added an explicit autograd fleet topology descriptor with discovered peer-performance ranks, generic-vs-native slot helpers, and a four-GPU generic reduction planner that adapts from peer-link quality while preserving the first slot as leader.
- Updated forward-neighbor device resolution so `generic` stays ordinal/topology-agnostic by default while `native` and `native-extreme` use the `0,2,1,3` order only after discovery confirms the native 4x V100 topology.
- Validated fresh `/tmp/cellerator-generic-build`, `/tmp/cellerator-native-build`, and `/tmp/cellerator-native-extreme-build` configure passes; built `computeAutogradRuntimeTest`, `forwardNeighborsCompileTest`, and `quantizedMatrixTest` successfully in all three trees; and ran those binaries successfully in all three trees.
- A follow-up build request for `seriesWorkbenchRuntimeTest` failed in those fresh trees because the target does not currently exist there, which appears to be a target-availability issue rather than a regression from the CUDA mode rewrite.
- Started the first-file freeze implementation: public docs/header comments now describe CSR/compressed as a legacy compatibility or interop path, the workbench runtime test no longer creates new compressed .csh5 fixtures, and cellShardDatasetH5Test now uses a Blocked-ELL side-domain append path plus a separate explicit legacy compressed compatibility check.
- Added a dedicated  target that writes a tiny optimized bucketed Blocked-ELL .csh5, warms cache and execution cache, summarizes the file, and reopens it to validate the frozen non-identity execution column remap.

## Next Actions
- Create or resume a workstream ledger under `todos/` for the next substantial task.
- Decide whether the first stable release excludes Python/H5AD by default unless that wheel and export app can be made reproducible quickly.
- Fix the installed include contract and rerun `cellShardInspectPackageTest` until it passes from a clean stage.
- Finish the downstream `part` to `partition` migration in the Cellerator targets that are supposed to stay green for the release matrix.
- Add the missing public-release basics: LICENSE, CI workflow, support matrix, and release notes.
- When release work is prioritized, start with the staged install contract in `extern/CellShard` before broader docs or packaging polish.
- Patch  and the package-consumer contract so the standalone install story is coherent.
- Choose and add the CellShard license file, then re-run the documented release checks if the license text changes packaging metadata or release notes.
- Review the final release candidate and decide when to tag .
- Review the final release candidate and decide when to tag `0.1.0`.
- Patch the runtime/workbench entrypoints first so Blocked-ELL `.csh5` files have a coherent consumer path before changing ingest planning and tests.
- Patch the CellShard series execution metadata and execution-cache path first, then update workbench blocked-ell consumers to use the new bucket-aware fetch helpers.
- Continue active workstreams; the CellShard file-surface rename stream is closed unless a broader API rename is requested.
- No further action in this stream unless the user wants a follow-up symbol/API vocabulary rename beyond the file-surface cleanup.
- Update includes/build/docs for , keep compatibility forwarding headers, and rerun the focused CellShard validation.
- Continue the remaining active workstreams; the CellShard directory-placement cleanup is complete unless a broader API vocabulary cleanup is requested.
- Continue the remaining active workstreams; the CellShard file-surface cleanup now has no legacy forwarding headers left.
- No further action in this stream unless the user wants additional naming cleanup such as `sharded/disk.cuh` or the remaining `series_h5` symbol vocabulary.
- Patch csh5 public structs and metadata load/write paths first, then update README wording and add a focused runtime-service metadata test.
- Plan the next code slice around owner-service orchestration: append staging, generation cutover, and pack rebuild/publish semantics without copying csh5 to executor nodes.
- Decide where the first concrete owner-node coordinator surface should live in CellShard beyond the new metadata contract.
- Implement the first concrete owner-node coordinator/runtime surface now that the operation model is documented explicitly.
- Start by converting the remaining compressed `.csh5` tests and deciding whether legacy compressed-file read support survives as a temporary compatibility layer.
- If compatibility is retained, make it explicitly read-only and time-bounded instead of preserving compressed write support.
- Finish the autograd fleet topology descriptor and slot-selection helpers, then update forward-neighbor device resolution and focused tests/bench consumers to use the new mode semantics.
- Decide whether to rename the remaining internal `portable_backend` / `extreme_backend` file vocabulary or keep it as implementation-only history, then continue the next benchmark-backed native-extreme specialization deeper into autograd or CellShard hot kernels.

## Done Criteria
- Every active workstream in `todos/` is reflected here with a current status.
- A clean standalone configure and build succeeds for the documented core scope.
- `cellShardExportRuntimeTest` and `cellShardInspectPackageTest` pass in a clean audit build.
- The agreed downstream Cellerator targets compile or run successfully against the release candidate CellShard revision.
- The release includes a license, versioned notes, documented supported build matrix, and automated checks for that matrix.
- Any deferred surfaces such as Python/H5AD are explicitly marked experimental or excluded rather than silently broken.
- Freshly converted `.csh5` series can be preprocessed and browsed through Blocked-ELL runtime paths without compressed fallbacks.
- Explicit warmup can populate shard packs into a caller-selected local cache root for a source file on local or remote storage.
- Persisted shard offsets are driven by runtime Blocked-ELL bytes plus CUDA `u32` caps, not ingest windows.
- Focused CellShard and workbench tests cover Blocked-ELL-only output, explicit cache warmup, and runtime reload/fetch behavior.
- The canonical CellShard sources use purpose-aligned filenames for the `.csh5` container and packfile codec responsibilities.
- Old include paths continue to compile through explicit compatibility forwarding headers or another intentional compatibility mechanism.
- CellShard build/test documentation points at the new canonical file names.
- A focused build succeeds after the rename.
- The public CellShard docs describe pack as the primary execution artifact served from the owner-hosted csh5 source.
- csh5 can persist and reload runtime-service and generation metadata needed for the owner-service contract.
- A focused test verifies the new metadata roundtrips through append/load and inspect accessors.
- The forward-looking CellShard file story no longer treats CSR/compressed as a native `.csh5` payload family.
- Any surviving CSR surface is clearly scoped to interop or legacy read-only compatibility and documented as such.
- Focused tests and docs no longer depend on creating new compressed `.csh5` datasets unless an intentional compatibility test remains.

## Historical Summary
- Recent completed work included Blocked-ELL persistence, real-data sparse replay benchmarking, quantize autograd kernels, workbench browse-cache updates, semantic cudaBioTypes cleanup, and the initial pointer-first neighbor workspace refactor.
- Detailed historical workstream notes are preserved in git history rather than as active repo ledgers.
dgers.
