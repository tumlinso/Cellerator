---
slug: "cellshard-first-stable-release"
status: "in_progress"
execution: "claimed"
owner: "codex"
created_at: "2026-04-13T15:49:29Z"
last_heartbeat_at: "2026-04-13T16:12:18Z"
last_reviewed_at: "2026-04-13T16:12:18Z"
stale_after_days: 14
objective: "bring CellShard to a first stable release"
---

# Current Objective

## Summary
Stabilize CellShard as a small standalone release by freezing scope around the C++/CUDA core and installable CMake package, fixing current package and downstream regressions, then adding the minimum docs, CI, and release metadata for v0.1.0.

## Quick Start
- Why this stream exists: CellShard already looks like a standalone library, but the release surface is not yet coherent enough to call stable.
- In scope: release scope definition, installed CMake package contract, downstream Cellerator compatibility, release docs and metadata, and a minimum green validation matrix.
- Out of scope / dependencies: no new storage formats, no broad runtime redesign, and no promise that the Python/H5AD surface ships in v0.1.0 unless its build becomes reliable.
- Required skills: `todo-orchestrator` for the ledger; use `cuda-v100` only if CUDA runtime packaging or Volta-specific runtime validation becomes the bottleneck.
- Required references: `extern/CellShard/README.md`, `extern/CellShard/CMakeLists.txt`, `extern/CellShard/pyproject.toml`, `extern/CellShard/src/CellShard.hh`, `extern/CellShard/src/sharded/sharded.cuh`, `src/models/dense_reduce/dR_dataloader.hh`, and `src/models/dense_reduce/dR_infer.hh`.

## Planning Notes
- Treat the first stable release as a narrow contract: a documented C++/CUDA storage and staging library with a working installed CMake package.
- Do not let optional Python export tooling define release readiness unless that surface is intentionally included in scope and validated end to end.
- Fixing the installed consumer contract is higher priority than adding more features because that contract is the release itself.

## Assumptions
- The first stable release includes the current Python/H5AD debugging and export surface rather than deferring it.
- The current `partition` naming in CellShard is the intended stable vocabulary; downstream code should migrate rather than forcing CellShard back to `part` names.
- A first stable tag should promise source-level compatibility for documented surfaces within `0.1.x`, not a broad ABI guarantee across arbitrary toolchains.

## Suggested Skills
- `todo-orchestrator` - Track the release plan and keep audit findings visible alongside active implementation work.

## Useful Reference Files
- `extern/CellShard/README.md` - Current stated scope of the standalone CellShard library.
- `extern/CellShard/CMakeLists.txt` - Defines the package exports, tests, install surface, and current version file.
- `extern/CellShard/pyproject.toml` - Defines the optional Python release surface and currently claims version 0.1.0.
- `extern/CellShard/src/CellShard.hh` - Umbrella include showing the intended public entry surface.
- `extern/CellShard/src/sharded/sharded.cuh` - Canonical partition and shard naming used by the current API.
- `src/models/dense_reduce/dR_dataloader.hh` - Concrete downstream code still using the pre-rename `part` API names.
- `src/models/dense_reduce/dR_infer.hh` - Additional downstream breakage against the renamed CellShard surface.

## Plan
- Freeze the release scope and decide whether v0.1.0 means core C++/CUDA only or also includes Python/H5AD export tooling.
- Fix the installed header and package contract so a clean staged consumer build passes reliably.
- Bring primary downstream Cellerator targets back to green against the current partition-named CellShard API.
- Add release hygiene: LICENSE, changelog or release notes, CI for the green matrix, and a documented support and dependency matrix.
- Cut and verify the release candidate from a clean build environment before tagging v0.1.0.

## Tasks
- [x] Decide and document the first stable release scope.
- [x] Make `cellShardInspectPackageTest` pass from a clean staged install.
- [x] Resolve the current Cellerator `part` to `partition` API drift.
- [x] Define and automate the release validation matrix.
- [~] Add legal and release metadata required for public distribution.
- [x] Harden the Python/H5AD packaging surface for the release path.

## Blockers
- A real release license is still missing. Choosing one without user approval is risky, so the code and packaging work is done but the legal/release metadata is not complete.

## Progress Notes
- Configured a standalone audit build with `cmake -S extern/CellShard -B /tmp/cellshard-release-audit -DCELLSHARD_BUILD_TESTS=ON -DCELLSHARD_ENABLE_PYTHON=OFF -DCELLSHARD_BUILD_APPS=OFF`.
- Built `cellShardExportRuntimeTest` successfully and ran it successfully in the audit build.
- Built the staged install/package-consumer path and confirmed `cellShardInspectPackageTest` fails when the consumer includes `CellShard/src/CellShard.hh` against exported include dirs rooted at `${prefix}/include/CellShard`.
- Built the root `modelCustomOpsTest` path and confirmed the current Cellerator downstream breakage is the `part` to `partition` API migration, not a packaging-only issue.
- Tried `python3 -m pip wheel extern/CellShard -w /tmp/cellshard-dist --no-deps` and observed the optional Python wheel fail while linking `cellshardH5adExport`.
- Claimed the release workstream for implementation; starting with install/package contract and Python linkage.
- Patched the standalone CellShard package contract so `cellShardInspectPackageTest` now validates both an inspect-only C++ consumer and a CUDA runtime consumer against the installed package.
- Linked `cellshard_inspect` transitively against `CUDA::cudart` in CUDA builds and passed the parent CUDA toolchain settings into the nested package-consumer configure step.
- Hardened Python packaging by removing the wheel-time dependency on the standalone `cellshardH5adExport` app, adding Python package classifiers, and exposing `cellshard.__version__`.
- Validated both `python3 -m pip wheel extern/CellShard -w /tmp/cellshard-dist --no-deps` and `python3 -m venv /tmp/cellshard-src-venv && /tmp/cellshard-src-venv/bin/pip install extern/CellShard` with successful `import cellshard` smoke checks.
- Migrated `src/models/dense_reduce/` to the current CellShard partition-named API and rebuilt `modelCustomOpsTest` successfully.
- Added release-facing basics inside `extern/CellShard`: `CHANGELOG.md`, `SUPPORT.md`, README support/package notes, and a starter CI workflow covering hosted CPU packaging plus self-hosted CUDA validation.
- Added  under Apache 2.0 and aligned the Python package metadata to ; reran the wheel build successfully.
- Added `extern/CellShard/LICENSE` under Apache 2.0 and aligned the Python package metadata to `license = "Apache-2.0"`; reran the wheel build successfully.

## Next Actions
- Choose and add the CellShard license file, then re-run the documented release checks if the license text changes packaging metadata or release notes.
- Review the final release candidate and decide when to tag .
- Review the final release candidate and decide when to tag `0.1.0`.

## Done Criteria
- A clean standalone configure and build succeeds for the documented core scope.
- `cellShardExportRuntimeTest` and `cellShardInspectPackageTest` pass in a clean audit build.
- The agreed downstream Cellerator targets compile or run successfully against the release candidate CellShard revision.
- The release includes a license, versioned notes, documented supported build matrix, and automated checks for that matrix.
- Any deferred surfaces such as Python/H5AD are explicitly marked experimental or excluded rather than silently broken.
