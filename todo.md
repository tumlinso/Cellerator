# Cellerator Todo

## Repository Picture

- `Cellerator` is aiming to be the single-cell pipeline and modeling layer on top of `CellShard`.
- The most real code today is in:
  - `src/ingest/series/*` and `src/ingest/mtx/*`
  - `src/compute/preprocess/*`
  - `bench/scrna_preprocess_bench.cu`
  - `tests/cellshard_series_h5_test.cu`
- The tree is still mid-migration:
  - old monolithic ingest code still exists in `src/load_embryo_mtx_shards.cu` and `src/load_embryo_series.cuh`
  - `src/rna/preprocess.cuh` is only a thin alias over `compute/preprocess`
  - `src/_rna/*`, `src/_models/*`, and `src/_apps/*` are mostly scaffold docs
  - `src/normalize.hh`, `src/normalize.cc`, and `src/targets/preprocess.cuh` are empty placeholders
- There is active local work in progress in this checkout, so cleanup should avoid broad churn until the intended migration path is explicit.

## What Looks Important Right Now

1. Make the build/test baseline trustworthy.
2. Choose the real ingest entrypoint and retire the duplicate one.
3. Turn the new preprocess path into a first-class pipeline target instead of just a benchmark/module.
4. Align the repo layout and docs with what is actually implemented.

## Immediate Next Steps

### 1. Fix build health first

- Replace or ignore the stale `build/` directory. It still points at `/home/tumlinson/Software/Cellerator`, not this checkout.
- Decide which CUDA toolchain is the supported one for this repo:
  - current configure tried `/usr/bin/nvcc`
  - benchmark notes reference the HPC SDK CUDA 12.9 toolchain
- Make `CMakeLists.txt` explicitly prefer the intended CUDA compiler/toolchain instead of relying on ambient environment state.
- Document a clean configure/build recipe in `README.md`.
- Re-run configure in a fresh directory and get at least one green compile target before doing more structural work.

### 2. Pick the active ingest path

- Decide whether `src/load_embryo_mtx_shards.cu` is still the real executable or whether `src/ingest/series/series_ingest.cuh` is the future center of gravity.
- If the modular path is the one to keep:
  - add a real executable target around `series_ingest`
  - move users and docs toward that path
  - mark the monolithic embryo loader as legacy
- If the monolith still matters short-term:
  - say so in the README
  - avoid implying that the modular series ingest has already replaced it

### 3. Promote preprocess from module to workflow

- Add a real compile/test target that exercises `src/compute/preprocess/*` directly.
- Cover the main contracts:
  - cell metrics
  - keep-cell mask
  - in-place normalize/log1p
  - gene metrics accumulation
  - gene filter mask
- Decide whether `src/rna/preprocess.cuh` should remain a compatibility alias or become the public API layer.
- Wire preprocess into an end-to-end path that can run on data loaded from `CellShard`, not only synthetic benchmark inputs.

### 4. Remove ambiguity in the repository structure

- Update `README.md` so the “current state” section matches the actual active code and tests.
- Either restore or intentionally drop the deleted `_ingest` scaffold docs.
- Move executable front-ends toward the advertised `_apps` layout, or stop promising that layout until the move starts.
- Mark empty placeholders clearly or delete them if they are only creating confusion:
  - `src/normalize.hh`
  - `src/normalize.cc`
  - `src/targets/preprocess.cuh`

## Near-Term Deliverables

### Ingest

- Finish the TSV manifest -> `series.csh5` conversion story around `src/ingest/series/series_ingest.cuh`.
- Add a small checked-in sample manifest plus tiny fixture data so ingest can be validated without ad hoc local files.
- Add failure-path coverage for:
  - missing barcodes/features
  - unsupported source formats
  - malformed TSV rows

### Preprocess

- Define a concrete “first supported preprocessing pass” for row-sharded CSR:
  - cell filtering
  - normalization/log1p
  - gene statistics
  - gene mask construction
- Decide what outputs should be persisted back into `CellShard` and what stays ephemeral.
- Add a correctness comparison against a simple CPU reference for at least one tiny matrix.

### Benchmarks and performance

- Keep `bench/scrna_preprocess_bench.cu`, but separate “performance benchmark” from “functional smoke test”.
- Capture one standard benchmark command line and expected output shape in docs.
- Make it obvious which benchmarks are about `CellShard` substrate performance versus `Cellerator` pipeline performance.

## Cleanup / Debt

- `tests/matrix_test.cc` still targets `src/matrix/*`, which is absent in this checkout. Either restore that code, move the test to `CellShard`, or delete the dead path.
- Tighten naming consistency:
  - `benchamarks.md` should likely become `benchmarks.md`
  - old `cudaHelpers.cuh` / `types.cuh` naming should be checked against the newer `support/` and `compute/` structure
- Decide which code belongs in `extern/CellShard` versus `src/compute/*` so reusable compute does not drift back into the storage layer.

## Questions To Resolve

- Is the first product milestone “ingest to `series.csh5`”, “GPU preprocessing on sharded CSR”, or both together?
- Is the repo targeting the system CUDA toolkit, the NVIDIA HPC SDK toolchain, or both?
- Should `Cellerator` keep legacy matrix-format code locally, or should anything storage-format-specific be pushed fully into `CellShard`?
- What is the smallest dataset fixture that should become the canonical end-to-end test input?

## Suggested Order

1. Get a clean configure/build working in a fresh build directory.
2. Choose and document the canonical ingest executable.
3. Add real preprocess correctness tests.
4. Connect ingest output to preprocess input.
5. Clean up placeholders and update docs to match reality.
