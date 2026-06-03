# CellStack Validation

This file records quick workspace-level validation commands and the expected
shape of their output. Shared fixture payloads live under `data/test/reference`;
generated outputs belong under `data/test/generated`.

## PBMC3K AnnData To CSH5 Ingest

Fixture pair:

- `data/test/reference/pbmc3k_raw.h5ad`
- `data/test/reference/pbmc3k_raw.csh5`

Build the CellShard comparison helper:

```bash
cmake -S CellShard -B CellShard/build -DCELLSHARD_BUILD_TESTS=ON -DCELLSHARD_BUILD_EXPORT=ON
cmake --build CellShard/build --target cellShardH5adCsh5AxisValidationTest -j 4
```

Run from the CellStack root:

```bash
./CellShard/build/cellShardH5adCsh5AxisValidationTest
```

The helper reads AnnData `X` through HDF5, materializes the `.csh5` through
CellShard CSR export, then summarizes shape, row pointer, column index, value,
row-sum, and column-sum deviations. The expected status is `PASS`; nonzero
deviations should be interpreted from the reported max, mean, total, and worst
row/column summaries.

Current result on the PBMC3K pair:

```text
shape h5ad: 2700 x 32738 nnz=2286884
shape csh5: 2700 x 32738 nnz=2286884
indptr mismatches: 0
value compared: 2286884
value column mismatches: 0
value mismatches: 0
value max_abs: 0
row_sum mismatches: 0
row_sum max_abs: 0
col_sum mismatches: 0
col_sum max_abs: 0
status: PASS
```

Default tolerances:

- absolute: `1e-3`
- relative: `1e-5`

Override paths or tolerances when needed:

```bash
./CellShard/build/cellShardH5adCsh5AxisValidationTest \
  --h5ad data/test/reference/pbmc3k_raw.h5ad \
  --csh5 data/test/reference/pbmc3k_raw.csh5 \
  --abs-tol 1e-3 \
  --rel-tol 1e-5
```
