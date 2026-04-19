---
slug: "cellshard-user-metadata-annotations"
status: "in_progress"
execution: "claimed"
owner: "codex"
created_at: "2026-04-15T16:09:11Z"
last_heartbeat_at: "2026-04-15T16:09:11Z"
last_reviewed_at: "2026-04-15T16:09:11Z"
stale_after_days: 14
objective: "replace hard-coded semantic metadata with cold user-defined observation/feature annotations and explicit owner-served fetch paths"
---

# Current Objective

## Summary
Implement the new metadata boundary for csh5: keep execution/runtime/provenance/browse built-in, move semantic observation/feature/preprocess metadata into cold user-defined annotation surfaces, and avoid shipping annotation payloads in default owner snapshots.

## Quick Start
- Why this stream exists: _Summarize the domain boundary and why it was split out._
- In scope: _List the work this stream owns._
- Out of scope / dependencies: _List handoffs, upstream dependencies, or adjacent streams._
- Required skills: _List the exact repo-local skills to read before starting._
- Required references: _List the exact repo-local references to read before starting._
- Read extern/CellShard/src/disk/csh5.cuh and extern/CellShard/src/disk/csh5.cc for the on-disk schema and append/load helpers before editing any metadata contract.
- Read src/workbench/dataset_workbench.hh, src/workbench/dataset_workbench.cc, and src/workbench/dataset_workbench_cuda.cu for dataset summaries, ingest metadata persistence, and preprocess finalization.
- Read extern/CellShard/export/dataset_export.hh, extern/CellShard/export/dataset_export.cc, and extern/CellShard/python/module.cc before changing owner/client metadata bootstrap or explicit fetch APIs.

## Planning Notes
_None recorded yet._

## Assumptions
_None recorded yet._

## Suggested Skills
- `todo-orchestrator` - Track the metadata schema work as a resumable repo-level workstream because it spans CellShard storage, workbench ingest, export/bootstrap, Python bindings, tests, and docs.

## Useful Reference Files
- `extern/CellShard/src/disk/csh5.cuh` - public csh5 schema structs and append/load entrypoints for metadata groups
- `extern/CellShard/src/disk/csh5.cc` - concrete HDF5 group names, writer helpers, and preprocess finalization path
- `src/workbench/dataset_workbench_cuda.cu` - current source metadata persistence and preprocess append logic that still hard-codes semantic day helpers and preprocess groups
- `extern/CellShard/export/dataset_export.cc` - owner snapshot serialization currently shipping observation metadata payloads by default

## Plan
_None recorded yet._

## Tasks
- [ ] Add feature annotation and dataset attribute metadata groups to the csh5 schema alongside observation annotations.
- [ ] Stop writing hard-coded semantic preprocess metadata into the built-in preprocess group and persist it through user-defined annotation surfaces instead.
- [ ] Keep user-defined metadata off the default owner snapshot and add explicit owner/client fetch helpers for annotation tables.
- [ ] Update H5AD/workbench ingest so obs and var annotations survive into the new user-defined metadata groups without day-specific hard coding.
- [ ] Refresh focused CellShard/workbench/runtime tests and docs for the new metadata contract.

## Blockers
_None recorded yet._

## Progress Notes
- Started implementation by auditing the current csh5 schema, workbench summary/loaders, preprocess finalize path, and owner snapshot serialization. The current blockers are that observation metadata is row-only, preprocess is a dedicated built-in group, and owner snapshots eagerly serialize full metadata payloads.

## Next Actions
- Patch the CellShard csh5 schema header and writer/reader helpers first so observation/feature annotations and dataset attributes have stable on-disk groups before touching ingest and Python APIs.

## Done Criteria
_None recorded yet._
