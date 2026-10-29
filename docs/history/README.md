# Preserved documentation

This directory keeps dated capability snapshots and paused design material out of the current product guide while preserving their original context. Begin with [references and history](../experiments_and_history.qmd) for current references and routes.

## Preserved CelleraTorch proposals

The two files in `celleratorch/` were moved from the root `CelleraTorch/` documentation-only workspace into the existing archive. Their original proposal scope is retained with its archive links corrected; [the archive index](celleratorch/README.md) explains their status. No package or build implementation moved with them.

## Historical implementation snapshot

[`current_implementation_2026-10-28.qmd`](current_implementation_2026-10-28.qmd) preserves the detailed dated record replaced by the concise current boundary at [`../current_implementation.qmd`](../current_implementation.qmd). The historical page includes an explicit note about the superseded “14 GPU checks” statement; the final frozen prepared-routing receipt reports 16 passing tests and exact CUDA/sanitizer checks.

## Ignored-document reconciliation

Before adoption, twelve ignored non-generated Markdown/Quarto files under `docs/` were preserved byte-for-byte by the cleanup checkpoint. The archive is `/mnt/block/cellerator/preservation/cleanup-checkpoint-20261029T061426Z/ignored-docs-20261029T062411Z.tar.zst`; its manifest records each path and checksum. Archive SHA-256: `5850e74ed53998cbe79868c04f01c78ff5560d12223f0f44313ff256d32e7bef`. The canonical ignored files remain present until the root adoption step.

Disposition: the old design overview's structure/discovery/realization principles are represented in [architecture](../architecture.qmd); the old getting-started and index pages are superseded by the current entry pages; `development/start.md` is superseded by the [developer reference](../developer_reference.qmd); the old tokenization proposal is superseded by the maintained [tokenization reference](../tokenization.qmd). The old source map is superseded by the current agent-facing [routing guide](../development/routing.md) and its generated map, which have separate ownership. The old status snapshot and results index are superseded by the current implementation and maintained performance/evidence references. The IR programming guide and cross-project program overview remain available in the verified backup archive; neither is promoted as current API guidance. AppleDouble `._*` files are metadata artifacts, not documentation.
