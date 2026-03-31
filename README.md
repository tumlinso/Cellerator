# Cellerator

Cellerator is the active single-cell bioinformatics pipeline in this repository.

It has two jobs:

1. provide the standard high-performance single-cell RNA toolkit pieces
2. host the Cellerator-specific modeling stack, especially state-based and quantized representations of cell state

`CellShard` is the storage substrate. `Cellerator` should read, write, stage, and operate on `CellShard` data, but the biological pipeline logic and model logic live here.

## Scope

`Cellerator` owns:

- dataset ingest and conversion
- preprocessing and normalization
- feature selection
- dimensional reduction
- nearest-neighbor search and graph construction
- clustering, embedding, and marker workflows
- low-level quantization code used by the pipeline
- Cellerator-specific models such as the state quantizer

`CellShard` owns:

- durable disk layouts
- sharded storage metadata
- packfile/container layout
- low-level persisted matrix formats

Short version:

- `Cellerator` transforms data
- `CellShard` stores data

## Current State

The current source tree is still early and matrix-heavy:

- `src/matrix/`: active sparse matrix formats, sharding, GPU staging, microscaled matrix experiments
- `src/load_embryo_mtx_shards.cu`: main ingest/conversion executable today
- `src/load_embryo_series.cuh`: large ingest path for embryo series work
- `src/normalize.*`: placeholder normalization slot

That is enough to build conversion and matrix tests, but it is not yet arranged like a general single-cell toolkit plus modeling stack.

## Target `src/` Layout

The intended long-term layout should separate infrastructure, standard RNA pipeline code, and Cellerator-specific modeling code.

Underscore rule:

- directories with a leading underscore are scaffolded but not yet migrated
- directories without a leading underscore are active homes for code that already lives there

Right now that means `src/matrix/` stays active, while the other top-level homes start as scaffold directories:

```text
src/
├── _apps/
│   ├── _convert/
│   ├── _inspect/
│   ├── _benchmark/
│   └── _verify/
├── _core/
│   ├── _cuda/
│   ├── _runtime/
│   ├── _types/
│   └── _util/
├── _ingest/
│   ├── _mtx/
│   ├── _h5ad/
│   └── _loom/
├── matrix/
│   ├── formats/
│   ├── sparse/
│   ├── sharded/
│   ├── convert/
│   ├── microscaled/
│   └── device/
├── _rna/
│   ├── _qc/
│   ├── _normalize/
│   ├── _features/
│   ├── _reduce/
│   ├── _neighbors/
│   ├── _graph/
│   ├── _cluster/
│   ├── _embed/
│   ├── _markers/
│   └── _pipeline/
└── _models/
    ├── _state_quantizer/
    ├── _states/
    ├── _objectives/
    ├── _train/
    └── _infer/
```

## Directory Roles

### `src/_apps/`

Command-line entry points only.

This is where current one-off executables should migrate over time.

Examples:

- `convert_embryo_mtx_shards.cu`
- `verify_packfile.cu`
- `benchmark_neighbors.cu`

### `src/_core/`

Project-wide low-level support code that is not specific to matrices or RNA.

This is where helpers such as CUDA utility code, shared scalar policy, small runtime helpers, and verification primitives should live.

### `src/_ingest/`

Dataset readers and conversion front-ends.

This layer should understand source file formats and how to turn them into working matrix views, but not contain the actual modeling logic.

For now, this layer should only cover:

- `_mtx`
- `_h5ad`
- `_loom`

Important split:

- `Cellerator` ingests source formats
- `CellShard` owns durable storage layout and container details

### `src/matrix/`

The reusable numerical substrate.

This holds:

- dense and sparse matrix layouts
- sharded views
- GPU upload/staging support
- sparse conversion kernels
- low-level microscaled matrix encodings

This is the right place for generic packed low-bit matrix layouts.

Examples:

- CSR/COO/CSC/DIA
- sharded matrix views
- microscaled CSR codebooks and dequantization helpers

### `src/_rna/`

All of the standard single-cell RNA pipeline work should live here.

This is the toolkit side of Cellerator.

Suggested subareas:

- `qc/`: cell and gene filtering, mitochondrial/ribosomal stats, count thresholds
- `normalize/`: library-size normalization, log transforms, residual normalization
- `features/`: HVG selection, gene masks, feature curation
- `reduce/`: PCA and related reductions
- `neighbors/`: exact/approximate kNN
- `graph/`: SNN, kNN graphs, graph utilities
- `cluster/`: Leiden/Louvain-style clustering
- `embed/`: UMAP or similar layouts
- `markers/`: marker scoring and DE-style summaries
- `pipeline/`: orchestration of standard RNA workflows

### `src/_models/`

This is the Cellerator-specific layer.

It should contain the new things that make the project distinct, not the generic toolkit pieces.

The first major resident here should be:

- `state_quantizer/`

This is where biologically meaningful state coding belongs: on/off-aware gene state estimation, code assignment policy, latent state construction, objectives, training loops, and inference rules.

Important distinction:

- generic low-level quantized matrix layouts belong in `src/matrix/microscaled/`
- biologically meaningful state quantization belongs in `src/_models/_state_quantizer/`

Both belong in `Cellerator`, but they should not be mixed together.

## Placement Rules

Use these rules when deciding where a new file should go:

1. If it is a reusable matrix/storage/kernel primitive, put it in `src/matrix/`.
2. If it is source-format ingest code for `mtx`, `h5ad`, or `loom`, put it in `src/_ingest/` until migrated.
3. If it is standard single-cell RNA workflow logic, put it in `src/_rna/` until migrated.
4. If it is specific to Cellerator’s biological modeling ideas, put it in `src/_models/` until migrated.
5. If it is only an executable front-end, put it in `src/_apps/` until migrated.

## Near-Term Migration

The tree does not need a big bang rewrite. The right move is gradual migration:

- keep `src/matrix/` as the active numerical base
- move standalone executables under `src/_apps/`
- move ingest-specific code under `src/_ingest/`
- build standard toolkit pieces under `src/_rna/`
- build the state quantizer under `src/_models/_state_quantizer/`

Concrete examples:

- `src/load_embryo_mtx_shards.cu` -> `src/_apps/_convert/embryo_mtx_to_cellshard.cu`
- `src/load_embryo_series.cuh` -> `src/_ingest/_mtx/embryo_series.cuh`
- `src/normalize.hh` / `src/normalize.cc` -> `src/_rna/_normalize/`

## Build

Current targets:

- `matrixTest`
- `convertEmbryoMtxShards`

As the tree grows, build targets should follow the new directory split:

- libraries or object groups for `core`, `matrix`, `ingest`, `rna`, `models`
- small executables under `src/_apps/`

## Direction

Cellerator should read like a serious single-cell toolkit with an attached research/modeling engine:

- strong standard RNA pipeline support
- direct compatibility with `CellShard`
- room for novel Cellerator-specific modeling tools
- clear separation between toolkit code and model code

That is the layout to optimize for.
