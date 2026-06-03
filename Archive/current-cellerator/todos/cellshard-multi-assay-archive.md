---
status: done
execution: closed
owner: codex
updated: 2026-04-28
---

# CellShard Multi-Assay Archive

## Quick Start
Implemented the first multi-assay storage foundation:

- `CellShard/io/multi_assay.hh` defines pointer-first assay semantics, row maps, pairing validation, paired row resolution, and per-assay pack manifest records.
- `.cshard` POD spec now reserves assay, pairing, row-map, and assay-pack descriptors without changing CSPACK payload codecs.
- Cellerator validation maps CellShard numeric semantics back to `the biology semantics package`.

## Suggested Skills
- `bio-experiments`
- `cuda`

## Useful References
- `the biology semantics package/src/omics/layout.hh`
- `CellShardinclude/CellShard/io/multi_assay.hh`
- `include/Cellerator/support/multi_assay_validation.hh`

## Done Criteria
- RNA+ATAC exact and partial pairing are validated against `the biology semantics package`.
- Missing-modality row lookup uses the CellShard invalid-row sentinel.
- CSPACK remains single-assay and is represented by metadata, not payload changes.
- Nearby tests build and pass.

## Progress Notes
- Built and ran `multiAssaySemanticsTest`.
- Built and ran `the biology semantics packageSemanticsTest`.
- Built and ran `cellShardExportRuntimeTest` to cover the `.cshard` export library after the header change.
