---
slug: "cellshard-first-class-build-export-python"
status: "stale"
execution: "closed"
owner: "codex"
created_at: "2026-04-13T14:45:32Z"
last_heartbeat_at: "2026-04-13T14:48:10Z"
last_reviewed_at: "2026-04-13T14:48:10Z"
stale_after_days: 14
objective: "promote CellShard to a first-class build target with optional export and python package surfaces"
stale_reason: "User marked this stream stale; keep it out of pickup rotation until explicitly reactivated."
---

# Current Objective

## Summary
Implement the CellShard split without changing current Cellerator behavior, add an optional export subtree for AnnData/H5AD conversion, add optional pybind and wheel packaging, and keep V100 runtime paths Blocked-ELL-first and pointer-first.

## Quick Start
_None recorded yet._

## Planning Notes
- Extend this workstream with a standalone CellShard no-GPU mode that keeps CUDA as the default and builds a separate host-only inspect/materialize path rather than adding fast-path fallbacks.

## Assumptions
- CellShard stays vendored under extern/CellShard for now and becomes separately buildable before any repo extraction.
- Default builds remain Cellerator-first and Python-free; export and Python targets are opt-in.
- H5AD export is a one-shot interoperability path and may widen values or indices at the boundary without changing native execution payloads.
- The no-GPU requirement applies to standalone extern/CellShard first; the root Cellerator build can remain CUDA-required in this patch.
- Host-only mode should support metadata inspection and small CPU materialization/export with explicit slow-path warnings, but not device staging or distributed runtime helpers.

## Suggested Skills
- `todo-orchestrator` - Keep the new CellShard split workstream current as implementation proceeds.
- `cuda-v100` - Keep sparse runtime and layout changes aligned to sm_70 V100 Blocked-ELL-first execution.

## Useful Reference Files
- `extern/CellShard/README.md` - Defines the intended CellShard scope and the current Blocked-ELL-first posture.
- `optimization.md` - Documents the repo-wide V100 bottlenecks and Blocked-ELL-first execution priorities.
- `pointer_migration_plan.md` - Constrains hot-path code away from vector-heavy GPU-facing abstractions.

## Plan
- Expand extern/CellShard/CMakeLists.txt into a real subproject with library targets, optional export target, optional pybind target, and install/export rules.
- Refactor the root CMake build to consume CellShard targets instead of recompiling CellShard source files into multiple executables.
- Add CellShard export core code under extern/CellShard/export for summary loading, CSR materialization, and Python-backed H5AD writing.
- Add optional pybind and wheel packaging under extern/CellShard/python while keeping the module out of the default build path.
- Add shard and part u32 limit validation in series_h5 creation paths, with hard errors for parts and warnings for oversized shards.
- Add focused runtime coverage for the export core and verify the default Cellerator build still configures and compiles.

## Tasks
- [x] Add a CellShard build switch that disables CUDA language/toolkit requirements and builds a host-only inspect library.
- [x] Introduce a host-only CUDA compatibility header so the inspect/materialize path can compile without nvcc.
- [ ] Add focused standalone no-GPU validation coverage and package export handling for CellShard::inspect.

## Blockers
- Standalone no-GPU build verification is currently blocked by existing series_h5.* symbol redefinitions during cellshard_inspect compilation, which overlaps the claimed cellshard-packfile-cache-rewrite stream.

## Progress Notes
- Confirmed the current CellShard standalone build hard-requires CUDA via project(... LANGUAGES CXX CUDA), find_package(CUDAToolkit REQUIRED), and CUDA-contaminated host-side headers.
- Confirmed the no-GPU runtime boundary can stay narrow: series_h5, shard_paths, disk, summary loading, fetch_part, and CSR/export materialization are host-side in behavior, while sharded_device, distributed, bucket, and convert kernels remain CUDA-only.
- Implemented CELLSHARD_ENABLE_CUDA with a host-only CellShard::inspect target, conditional CellShard::runtime export, and conditional CUDAToolkit package dependency so standalone CellShard now configures without nvcc when CUDA is disabled.
- Added src/cuda_compat.cuh plus shared-header cleanup so summary/fetch/materialize/export code compiles as pure C++ in no-GPU mode while the default CUDA build still exposes the existing GPU-facing headers.
- Validated standalone no-GPU configure/build/test and package export with /tmp/cellshard_nogpu_build plus a clean consumer that links CellShard::inspect, then revalidated the default CUDA-enabled standalone export test in /tmp/cellshard_cuda_build.
- Claimed this idle stream and narrowed the next patch to standalone package-component correctness for no-GPU CellShard installs.
- Patched CellShardConfig.cmake.in to validate requested package components against exported targets, added a package-consumer smoke project, and documented the standalone component contract in extern/CellShard/README.md.

## Next Actions
- Implement the standalone CellShard subproject targets and rewire the root build to link them.
- Patch extern/CellShard/CMakeLists.txt, CellShardConfig.cmake.in, and the shared headers to introduce the standalone no-GPU inspect/materialize mode without disturbing the default CUDA runtime path.
- If the broader split continues, rewire the root Cellerator build to consume CellShard::inspect and CellShard::runtime explicitly instead of assuming one monolithic CellShard runtime target.
- Patch CellShardConfig.cmake.in to validate requested components against exported targets, then add focused standalone no-GPU package-consumer coverage.
- After the overlapping series_h5 redefinition issue is resolved, rerun standalone extern/CellShard with CELLSHARD_ENABLE_CUDA=OFF and build the new cellShardInspectPackageTest target.

## Done Criteria
- CellShard configures as a standalone project and the root Cellerator build consumes its targets without behavior regressions.
- Optional export and Python packaging surfaces exist without becoming mandatory dependencies of the default build.
- Series creation emits part hard-limit failures and shard limit warnings for u32-sensitive execution boundaries.
- Standalone CellShard configures and builds with CELLSHARD_ENABLE_CUDA=OFF on a machine without nvcc, exposing inspect/materialize/export surfaces but not GPU runtime targets.
- The default standalone CUDA-enabled CellShard build still produces the current runtime targets and preserves the existing V100-oriented hot path behavior.
