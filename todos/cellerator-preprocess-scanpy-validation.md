---
slug: "cellerator-preprocess-scanpy-validation"
status: "done"
execution: "closed"
owner: "codex"
created_at: "2026-05-06T00:00:00Z"
last_heartbeat_at: "2026-05-06T00:00:00Z"
last_reviewed_at: "2026-05-06T00:00:00Z"
stale_after_days: 3
objective: "Validate Cellerator GPU-native preprocessing metrics against Scanpy on the PBMC3K h5ad/csh5 fixture pair."
---

# Current Objective

## Summary
Validate the Cellerator `.csh5` preprocessing session against a Scanpy reference
for `data/test/reference/pbmc3k_raw.h5ad` and
`data/test/reference/pbmc3k_raw.csh5`.

## Quick Start
- In scope: Cellerator-owned metric exposure, reusable Scanpy comparison script,
  and focused PBMC3K validation.
- Out of scope: full persisted filtered/normalized matrix publication readback.
- Required skills: `bio-experiments` for scRNA preprocessing semantics.
- Required references: `AGENTS.md`, `tests/python_preprocess_smoke.py`,
  `python/module.cc`, and `python/cellerator/pp.py`.

## Planning Notes
- Treat PBMC3K as scRNA raw counts with observations by features.
- Use Scanpy/SciPy only for the reference path; Cellerator preprocessing stays
  GPU-native and `.csh5`-first.

## Assumptions
- `pbmc3k_raw.h5ad` and `pbmc3k_raw.csh5` describe the same feature order and
  observation order.
- The first validation pass compares QC metrics, keep masks, and normalized
  gene reductions, not full matrix publication.

## Suggested Skills
- `bio-experiments`: preserve raw-count and normalization semantics.

## Useful Reference Files
- `tests/validate_scanpy_preprocess.py`: reusable PBMC3K validation script.
- `tests/python_preprocess_smoke.py`: Python import and fixture smoke pattern.
- `python/module.cc`: Python metric exposure and session binding.

## Plan
- Expose missing already-collected session metrics to Python.
- Add a Scanpy reference comparator for the PBMC3K `.h5ad`/`.csh5` pair.
- Build the Python-enabled Cellerator target and run focused validation.
- Record validation results and any remaining tolerance/fidelity findings.

## Tasks
- [x] Expose Cellerator session metrics needed for Scanpy comparison.
- [x] Add reusable Scanpy validation script.
- [x] Run Python-enabled Cellerator build.
- [x] Run existing Python preprocessing smoke test with PBMC3K `.csh5`.
- [x] Run Scanpy comparison script against the PBMC3K fixture pair.
- [x] Record validation outcome.

## Blockers
_None recorded yet._

## Progress Notes
- Added the workstream while implementing the PBMC3K validation plan.
- Fixed the Python binding's direct session path to reserve CellShard device
  partition records before upload; the C++ session path already did this.
- Fixed the Cellerator session `cell_mito_counts` alias so it copies the
  mitochondrial group count per row rather than the first contiguous group
  metric block.
- Validation passed:
  `cmake -S Cellerator -B Cellerator/build -DCELLERATOR_ENABLE_PYTHON=ON -DCELLERATOR_BUILD_PREPROCESS_WORKBENCH=OFF -DCELLERATOR_BUILD_PREPROCESS_BENCHMARKS=OFF -DCELLERATOR_ENABLE_TORCH_MODELS=OFF`
- Validation passed: `cmake --build Cellerator/build -j 4`
- Validation passed:
  `CELLERATOR_PREPROCESS_CSH5=../data/test/reference/pbmc3k_raw.csh5 ./build/celleratorPreprocessCellShardSessionApiTest`
- Validation passed:
  `PYTHONPATH=Cellerator/python python Cellerator/tests/python_preprocess_smoke.py --build-dir Cellerator/build --dataset data/test/reference/pbmc3k_raw.csh5`
- Validation passed:
  `PYTHONPATH=Cellerator/python python Cellerator/tests/validate_scanpy_preprocess.py --build-dir Cellerator/build --h5ad data/test/reference/pbmc3k_raw.h5ad --csh5 data/test/reference/pbmc3k_raw.csh5`
- PBMC3K comparison passed for shape, feature group masks, cell totals,
  detected genes, mitochondrial counts, max counts, group counts/percentages,
  cell keep mask, normalized-log1p gene sums/squared sums, gene detected cells,
  and gene keep mask.

## Next Actions
_None._

## Done Criteria
- The PBMC3K fixture comparison runs from a clean command and reports pass/fail
  for each metric family.
- Any numerical tolerance or data-layout mismatch is recorded clearly.
