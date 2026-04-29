# CellShardPreprocess GPU Biology Backbone

Last updated: 2026-04-26

## Quick Start

Migrate the accelerated scRNA preprocessing and standard biology backbone from Cellerator into `extern/CellShardPreprocess`. Use `bio-experiments` for preprocessing semantics and `cuda` native V100 routing for GPU layout decisions. Preserve Cellerator C++ and Python APIs as wrappers that delegate to CellShardPreprocess.

## Status

- Status: done
- Execution: closed
- Owner: none

## Assumptions

- Assay is scRNA-seq, with explicit raw-count input state before QC, normalize-total, and log1p.
- Matrix orientation and feature semantics must stay explicit at API boundaries.
- CellShardPreprocess treats Blocked-ELL and Sliced-ELL as peer first-class preprocessing layouts; compressed / CSR is fallback.
- CellShardPreprocess may depend directly on CellShard runtime/export/internal sparse definitions.
- No compatibility alias is required for the previous `CellSlinger` shell.
- The target Git remote is `git@github.com:tumlinso/CellShardPreprocess.git`.

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
- [x] Rename or create `extern/CellShardPreprocess`, update Git submodule metadata and remote.
- [x] Stand up standalone CellShardPreprocess CUDA/C++ build with smaller `CellShardPreprocess::preprocess` and composed `CellShardPreprocess::runtime` targets, namespace `cellshard_preprocess`, Python package `cellshard_preprocess`, and optional `cellShardPreprocessWorkbench`.
- [x] Move accelerated preprocessing behavior into CellShardPreprocess: raw-count validation, cell/gene QC metrics, mitochondrial flags, normalize-total/log1p, preprocessing metadata hooks, and compaction/finalization integration points.
- [x] Remove Cellerator preprocessing compatibility wrappers and build targets after moving ownership to CellShardPreprocess.
- [x] Add focused CellShardPreprocess-native and Cellerator compatibility tests.
- [x] Update documentation surfaces affected by the migration.
- [x] Verify standalone CellShardPreprocess, integrated Cellerator build, and runtime tests.

## Progress Notes

- Ledger initialized from the supplied plan.
- Renamed the external shell to `extern/CellShardPreprocess` and updated the submodule URL to `git@github.com:tumlinso/CellShardPreprocess.git`.
- Added pointer-first CellShardPreprocess native preprocessing headers, CUDA implementation, Python package marker, standalone CMake target, host tests, GPU preprocessing tests, and optional `cellShardPreprocessWorkbench`.
- Split CellShardPreprocess into the smaller `CellShardPreprocess::preprocess` target and the composed `CellShardPreprocess::runtime` target so native preprocessing can build before Cellerator links it.
- Added standalone CellShard resolution for cloned CellShardPreprocess checkouts: explicit `CELLSHARD_PREPROCESS_CELLSHARD_SOURCE_DIR`, sibling checkout, vendored checkout, then `FetchContent`.
- Linked Cellerator workbench against `CellShardPreprocess::runtime` and routed raw-count/double-preprocess validation through CellShardPreprocess.
- Completed the final migration pass: CellShardPreprocess now exposes Blocked-ELL and Sliced-ELL as first-class preprocessing APIs, keeps compressed / CSR as fallback, owns the moved preprocessing benchmarks, and Cellerator no longer exposes preprocessing headers or root preprocessing benchmark targets.
- Fixed stale benchmark build issues encountered during the required full Cellerator build.
- Verified standalone CellShardPreprocess configure/build, full Cellerator build, host-runnable CellShardPreprocess tests, and existing Cellerator compatibility tests.

## Blockers

- GPU runtime tests build but could not run in this environment because CUDA reported no device.

## Next Actions

- No migration action remains. Future work should benchmark Blocked-ELL and Sliced-ELL preprocessing as peer native CellShardPreprocess layouts before choosing a preferred runtime format.

## Done Criteria

- `extern/CellShardPreprocess` exists with the requested remote metadata and standalone build.
- Cellerator builds without public preprocessing APIs or root preprocessing benchmark targets.
- CellShardPreprocess-native tests cover CellShard-format preprocessing, adapter staging, QC equivalence, double-preprocess rejection, and workbench compile coverage where feasible.
- Documentation reflects the new ownership boundary.
- No Cellerator preprocessing compatibility code remains; native preprocessing, metadata persistence contracts, compaction/finalization hooks, and browse-cache support live under CellShardPreprocess/CellShard-owned targets.
