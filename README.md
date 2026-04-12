# Cellerator

Cellerator is the compute and model layer on top of `CellShard`.

`CellShard` owns persisted matrix layout, staging, and fetch/drop primitives. `Cellerator` owns ingest, preprocessing, reusable sparse compute, model code, trajectory logic, the workbench surface, and the quantized runtime.

This repo is intentionally low-level and V100-oriented. Hot paths should stay explicit about layout, residency, transfers, and launch structure rather than being hidden behind generic abstractions.

## Layout

Active top-level code lives under:

- `src/compute/`: reusable compute, preprocess, neighbors, model ops, and sparse autograd building blocks
- `src/ingest/`: source-format readers, MTX conversion, and series ingest
- `src/workbench/`: ncurses workbench and related orchestration helpers
- `src/models/`: header-first libtorch workflows
- `src/trajectory/`: trajectory scoring, pruning, and assembly
- `src/quantized/`: low-level quantized sparse backend
- `src/torch/`: explicit Torch export boundary
- `src/support/`: shared small support headers
- `src/legacy/`: quarantined monolithic or transitional code

Native sparse posture:

- Blocked-ELL is the preferred persisted and execution layout
- compressed / CSR remains a fallback and interop path where still required

## Build

Configure and build with:

```bash
cmake -S . -B build
cmake --build build -j 4
```

Useful targets include:

- `scrnaPreprocessBench`
- `cellShardSeriesH5Test`
- `quantizedMatrixTest`
- `seriesIngestCompileTest`
- `seriesWorkbenchRuntimeTest`
- `celleratorWorkbench`
- `trajectoryCompileTest`
- `trajectoryRuntimeTest`
- `forwardNeighborsCompileTest`
- `computeAutogradRuntimeTest`
- `developmentalTimeCompileTest`
- `denseReduceCompileTest`
- `quantizeModelTest`
- `torchBindingsCompileTest`
- `modelCustomOpsTest`

Torch-enabled builds prefer the source-built libtorch under `/usr/local/share/cmake/Torch`. If libtorch is unavailable:

```bash
cmake -S . -B build -DCELLERATOR_ENABLE_TORCH_MODELS=OFF
```

Only override with `Torch_DIR` or `LIBTORCH_PATH` when testing against a different libtorch build.

## Testing

`ctest` is not configured. Run built binaries directly, for example:

```bash
./build/cellShardSeriesH5Test
./build/forwardNeighborsCompileTest
./build/computeAutogradRuntimeTest
./build/quantizeModelTest
```

When touching GPU-facing code, record the exact build and run commands used.

## References

- `AGENTS.md`: repository-specific engineering rules
- `todos.md`: active planning ledger
- `optimization.md`: current measured performance guidance
- `pointer_migration_plan.md`: pointer-first migration policy for hot paths
- `custom_torch_ops.md`: model-facing custom op registry
