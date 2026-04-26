# MosaiCell GPU Biology Backbone

Last updated: 2026-04-26

## Quick Start

Migrate the accelerated scRNA preprocessing and standard biology backbone from Cellerator into `extern/MosaiCell`. Use `bio-experiments` for preprocessing semantics and `cuda` native V100 routing for GPU layout decisions. Preserve Cellerator C++ and Python APIs as wrappers that delegate to MosaiCell.

## Status

- Status: partial
- Execution: active
- Owner: current Codex thread

## Assumptions

- Assay is scRNA-seq, with explicit raw-count input state before QC, normalize-total, and log1p.
- Matrix orientation and feature semantics must stay explicit at API boundaries.
- MosaiCell may depend directly on CellShard runtime/export/internal sparse definitions.
- No compatibility alias is required for the previous `CellSlinger` shell.
- The target Git remote is `git@github.com:tumlinso/MosaiCell.git`.

## References

- `AGENTS.md`
- `optimization.md`
- `docs/pipeline/preprocess/README.md`
- `docs/pipeline/README.md`
- `bio-experiments/references/assay-scrna.md`
- `bio-experiments/references/task-preprocessing.md`
- `cuda/references/systems/native.md`
- `cuda/references/addendum-bio-data-layouts.md`

## Plan

- [x] Inspect existing `extern`, preprocessing, ingest, workbench, Python, and CMake surfaces.
- [x] Rename or create `extern/MosaiCell`, update Git submodule metadata and remote.
- [x] Stand up standalone MosaiCell CUDA/C++ build with smaller `MosaiCell::preprocess` and composed `MosaiCell::runtime` targets, namespace `mosaicell`, Python package `mosaicell`, and optional `mosaiCellWorkbench`.
- [ ] Move or wrap accelerated preprocessing behavior in MosaiCell: raw-count validation, cell/gene QC metrics, mitochondrial flags, normalize-total/log1p, preprocessing metadata, compaction/finalization, and browse-cache support.
- [x] Update Cellerator compatibility wrappers and build targets so existing APIs delegate to MosaiCell.
- [x] Add focused MosaiCell-native and Cellerator compatibility tests.
- [x] Update documentation surfaces affected by the migration.
- [x] Verify standalone MosaiCell, integrated Cellerator build, and runtime tests.

## Progress Notes

- Ledger initialized from the supplied plan.
- Renamed the external shell to `extern/MosaiCell` and updated the submodule URL to `git@github.com:tumlinso/MosaiCell.git`.
- Added pointer-first MosaiCell native preprocessing headers, CUDA implementation, Python package marker, standalone CMake target, host tests, GPU preprocessing tests, and optional `mosaiCellWorkbench`.
- Split MosaiCell into the smaller `MosaiCell::preprocess` target and the composed `MosaiCell::runtime` target so native preprocessing can build before Cellerator links it.
- Added standalone CellShard resolution for cloned MosaiCell checkouts: explicit `MOSAICELL_CELLSHARD_SOURCE_DIR`, sibling checkout, vendored checkout, then `FetchContent`.
- Linked Cellerator workbench against `MosaiCell::runtime` and routed raw-count/double-preprocess validation through MosaiCell.
- Preserved the existing Cellerator workbench facade for compatibility rather than copying its vector-heavy public shape into MosaiCell. The existing compatibility implementation still contains Cellerator-owned metadata, finalize, browse-cache, compressed fallback, and sliced-ELL paths that need a deeper extraction before this migration is complete.
- Fixed stale benchmark build issues encountered during the required full Cellerator build.
- Verified standalone MosaiCell configure/build, full Cellerator build, host-runnable MosaiCell tests, and existing Cellerator compatibility tests.

## Blockers

- GPU runtime tests build but could not run in this environment because CUDA reported no device.

## Next Actions

- Move the remaining Cellerator preprocess orchestration and metadata/finalize/browse-cache support behind MosaiCell-owned pointer-first APIs without copying the vector-heavy compatibility facade.
- Keep compressed/CSR and sliced-ELL Cellerator paths explicitly marked as fallback or migration surfaces while MosaiCell's Blocked-ELL-first path becomes the native owner.

## Done Criteria

- `extern/MosaiCell` exists with the requested remote metadata and standalone build.
- Cellerator builds with MosaiCell integrated and keeps existing preprocessing/workbench APIs working.
- MosaiCell-native tests cover CellShard-format preprocessing, adapter staging, QC equivalence, double-preprocess rejection, and workbench compile coverage where feasible.
- Documentation reflects the new ownership boundary.
- Remaining Cellerator compatibility code is thin wrapper/orchestration only; native preprocessing, metadata persistence contracts, compaction/finalization hooks, and browse-cache support live under MosaiCell-owned targets.
