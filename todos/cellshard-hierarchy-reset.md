---
slug: "cellshard-hierarchy-reset"
status: "in_progress"
execution: "claimed"
owner: "codex"
created_at: "2026-04-18T13:39:44Z"
last_heartbeat_at: "2026-04-18T13:39:44Z"
last_reviewed_at: "2026-04-18T13:39:44Z"
stale_after_days: 14
objective: "Reorganize CellShard into a layered hierarchy with a curated public include tree, without splitting hot kernels"
---

# Current Objective

## Summary
Reset CellShard around a layered source tree and a curated public include surface, then migrate in-repo consumers onto that public tree while leaving fused kernels intact.

## Quick Start
- Why this stream exists: _Summarize the domain boundary and why it was split out._
- In scope: _List the work this stream owns._
- Out of scope / dependencies: _List handoffs, upstream dependencies, or adjacent streams._
- Required skills: _List the exact repo-local skills to read before starting._
- Required references: _List the exact repo-local references to read before starting._
- Why this stream exists: CellShard is still structurally flat for agentic work; large mixed-responsibility files and broad umbrellas force more context than necessary.
- In scope: public include/install contract, `src/core`/`src/io`/`src/runtime` hierarchy, export and python splits, downstream include migration, and focused validation.
- Out of scope: kernel splitting, fused-kernel decomposition, and algorithm changes in hot paths.

## Planning Notes
- The earlier `split-csh5-translation-units` stream preserved behavior but intentionally stopped short of a semantic hierarchy reset.
- The installed/public include contract currently exports almost the whole `src/` tree, while Cellerator reaches directly into many deep CellShard paths.

## Assumptions
- A curated public include tree under `include/CellShard/` is the cleanest stable surface; internal source layout may change freely behind it.
- Existing downstream includes in Cellerator may be updated broadly in the same pass instead of preserving old paths.

## Suggested Skills
- `todo-orchestrator` - Track this repo-wide refactor as a dedicated resumable workstream.

## Useful Reference Files
- `extern/CellShard/CMakeLists.txt` - Owns the current public install and include contract.
- `extern/CellShard/src/CellShard.hh` - Current oversized umbrella surface to replace with a thin curated umbrella.
- `extern/CellShard/export/dataset_export.hh` - Public export API surface that will drive the new export hierarchy.

## Plan
- Create the new public include tree and update CellShard CMake/install rules to publish only that curated surface.
- Move non-kernel CellShard internals into layered `src/core`, `src/io`, and `src/runtime` directories and update internal includes.
- Split export and python into smaller domain files that depend on shared helpers instead of copy-pasted local utilities.
- Migrate CellShard tests, package consumers, and Cellerator include sites onto the new public headers, then rebuild focused targets.

## Tasks
- [x] Define and create the new public include tree.
- [x] Rehome CellShard non-kernel internal files into layered directories.
- [x] Split export and python surfaces by behavior.
- [x] Update CMake/install/package-consumer coverage.
- [x] Rebuild and run focused CellShard and Cellerator validation targets.

## Blockers
_None recorded yet._

## Progress Notes
- Created a dedicated workstream for the CellShard hierarchy reset so it does not blur with the earlier translation-unit split or the release stream.
- Landed the curated `include/CellShard/` public tree and rewired CellShard install/package-consumer coverage onto that surface.
- Rehomed core, io, and runtime implementation files under layered `src/core`, `src/io`, and `src/runtime` directories while leaving hot kernels intact.
- Split the Python binding surface into `python/module.cc`, `python/bind_types.cc`, `python/bind_handles.cc`, and shared helpers under `python/internal/`.
- Replaced the old monolithic `export/dataset_export.cc` with `export/summary/load_summary.cc`, `export/materialize/csr_export.cc`, and `export/snapshot/global_metadata_snapshot.cc`, backed by shared internal helpers.
- Split the old `src/io/csh5/runtime.cc` blob into `src/io/csh5/runtime/{bind,headers,fetch,warm}.cc` with shared runtime helpers, and removed the old `src/io/csh5/internal.hh` funnel in favor of narrower private wrappers.
- Focused validation is green for the export/runtime path, the package-consumer install surface, and the top-level dataset/workbench runtime targets.

## Next Actions
- If a follow-on cleanup is needed, target the large helper part files under `src/io/csh5/*_part.hh`, which are now behind narrower private wrappers but still hold the deepest private implementation density.

## Done Criteria
- CellShard installs and documents only the curated public headers.
- Downstream Cellerator code no longer includes private CellShard helper paths.
- Export and python are split into smaller behavior-local files.
- Fused kernels remain intact and focused validation passes after the migration.
