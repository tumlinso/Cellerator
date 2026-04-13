---
slug: "rewrite-cudabiotypes-semantic-contracts"
status: "done"
execution: "closed"
owner: "unassigned"
created_at: "2026-04-13T14:47:07Z"
last_heartbeat_at: "2026-04-13T14:47:07Z"
last_reviewed_at: "2026-04-13T14:47:07Z"
stale_after_days: 14
objective: "rewrite cudabiotypes semantic contracts"
---

# Current Objective

## Summary
Rewrite extern/cudaBioTypes into a meaning-only biology contract layer, then add a Cellerator-side semantic validator that checks CellShard matrices against assay specs without changing runtime ownership.

## Quick Start
_None recorded yet._

## Planning Notes
- cudaBioTypes now owns biological meaning only: modality, feature/observation identity, provenance, processing state, and pairing semantics.
- Runtime sparse layout ownership, blocked-ELL policy, persistence, fetch/drop, and execution remain in CellShard or Cellerator.

## Assumptions
- Primary assay coverage is single-cell-first: scRNA, scATAC, CITE-seq ADT, spatial, and matched multimodal single-cell.
- Semantics should be metadata-first and may use lightweight host containers, but must not become a new hot-path runtime matrix abstraction.
- Cellerator integration should validate meaning at entry boundaries rather than wrap every internal operator.

## Suggested Skills
- `bio-experiments` - Keep assay, processing-state, and pairing semantics scientifically correct.
- `cuda-v100` - Keep sparse-format and runtime ownership decisions on the sm_70 execution side, not in the semantic layer.

## Useful Reference Files
- `extern/cudaBioTypes/docs/scope.md` - Declares cudaBioTypes as the biology-facing type layer beneath CellShard and Cellerator.
- `pointer_migration_plan.md` - Prevents semantic integration from introducing new vector-heavy runtime APIs in hot code.
- `/home/tumlinson/.agents/skills/bio-experiments/references/hard-constraints.md` - Defines modality, provenance, and processing-state guardrails for assay semantics.

## Plan
- Trim extern/cudaBioTypes to semantic-only headers and docs; remove IO, sparse-layout, and CellShard-interop responsibilities from the public umbrella.
- Introduce metadata-first assay spec and multimodal linkage types that encode modality, axis meaning, feature namespace, processing state, and provenance.
- Wire cudaBioTypes into the root build as an interface dependency and add a Cellerator-side validator for CellShard compressed and blocked-ELL matrices.
- Add focused tests covering valid assay specs, invalid semantic combinations, and CellShard matrix/assay mismatches.

## Tasks
- [x] Rewrite extern/cudaBioTypes public API around meaning-only assay and schema contracts.
- [x] Remove stale IO and CellShard interop surfaces from extern/cudaBioTypes build and docs.
- [x] Add a Cellerator semantic validation header that checks CellShard matrices against cudaBioTypes assay specs.
- [x] Add or update tests for semantic validation in both extern/cudaBioTypes and Cellerator.

## Blockers
_None recorded yet._

## Progress Notes
- Rewrote cudaBioTypes around semantic assay contracts, multimodal pairing metadata, and axis/processing-state meaning; removed sparse layout ownership and IO/interop headers from the public surface.
- Added src/support/assay_validation.hh plus a root cudaBioTypesSemanticsTest target that validates CellShard compressed and blocked-ELL matrices against cudaBioTypes assay semantics without introducing a new runtime matrix owner.
- Verified the standalone cudaBioTypes smoke test and the root cudaBioTypesSemanticsTest target after narrowing CellShard includes to format-only headers.
- Workstream implementation is complete for the semantic-only cudaBioTypes rewrite and the initial Cellerator validation boundary.
- Added a stability pass that renames the public namespace to `bioT`, keeps a `cudaBioTypes` compatibility alias at the umbrella boundary, and updates docs to match the semantic-only surface actually checked into `extern/cudaBioTypes`.
- Removed the leftover alignment layer and public alignment-derived aliases so the semantic surface now excludes read, mapping, kmer, and partition types.

## Next Actions
- Patch cudaBioTypes headers and docs first so the semantic API is stable before wiring Cellerator integration.
- If broader adoption is needed, thread the semantic validator into concrete ingest/model entrypoints rather than adding a universal matrix wrapper.
- Use the new semantic validator at specific Cellerator entry boundaries when future tasks need meaning checks.

## Done Criteria
- cudaBioTypes public umbrella contains only semantic biology contracts and no IO or CellShard interop headers.
- Cellerator can compile a semantic validator against CellShard matrices and cudaBioTypes assay specs.
- Tests cover modality, processing-state, axis, and feature-type mismatches without introducing a new runtime matrix owner.
- The public namespace used by current consumers is `bioT`, and the compatibility alias does not break existing umbrella-header includes.
